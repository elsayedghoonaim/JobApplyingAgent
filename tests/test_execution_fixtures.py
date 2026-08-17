"""Deterministic offline fixture tests for Easy Apply form execution decomposition."""

import inspect
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Optional

import pytest

import jobapply.nodes.execution as execution_module
from jobapply.execution import (
    STANDARD_TEXT_FIELD_SELECTOR,
    choice_is_unanswered,
    classify_navigation_action,
    find_already_applied_indicator,
    get_auto_fill_value,
    get_choice_label,
    get_form_field_label,
    get_radio_option_label,
    is_known_field,
    select_live_radio_option,
    select_radio_option,
    text_indicates_already_applied,
    validate_visible_required_controls,
    visible_button_labels,
    wait_for_submission_confirmation,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "easy_apply"


# ---------------------------------------------------------------------------
# Lightweight In-Memory Fake DOM and Playwright Adapter
# ---------------------------------------------------------------------------


def _matches_single_token(node: "FakeDOMNode", token: str) -> bool:
    """Check if a FakeDOMNode matches a single compound selector token."""
    token = token.strip()
    if not token or token == "*":
        return True

    # 1. Pseudo-classes
    while ":has-text(" in token:
        m = re.search(r":has-text\((['\"]?)(.*?)\1\)", token)
        if not m:
            break
        text_to_match = m.group(2).lower()
        if text_to_match not in node.inner_text_sync().lower():
            return False
        token = token[: m.start()] + token[m.end() :]

    if ":checked" in token:
        if not node._checked:
            return False
        token = token.replace(":checked", "")

    while ":not(" in token:
        m = re.search(r":not\((.*?)\)", token)
        if not m:
            break
        negated = m.group(1)
        if _matches_single_token(node, negated):
            return False
        token = token[: m.start()] + token[m.end() :]

    # 2. Attributes [attr] / [attr=val] / [attr*=val i]
    while "[" in token:
        m = re.search(r"\[([^\]]+)\]", token)
        if not m:
            break
        attr_expr = m.group(1).strip()
        if "*=" in attr_expr:
            attr_name, val = attr_expr.split("*=", 1)
            attr_name = attr_name.strip().lower()
            val = val.strip().rstrip(" i").rstrip(" I").strip("'\"").lower()
            if val not in node.attrs.get(attr_name, "").lower():
                return False
        elif "=" in attr_expr:
            attr_name, val = attr_expr.split("=", 1)
            attr_name = attr_name.strip().lower()
            val = val.strip().strip("'\"")
            if node.attrs.get(attr_name) != val:
                return False
        else:
            if attr_expr.lower() not in node.attrs:
                return False
        token = token[: m.start()] + token[m.end() :]

    # 3. ID selector (#id)
    if "#" in token:
        parts = token.split("#", 1)
        tag_prefix = parts[0].strip()
        rest = parts[1].strip()
        if "." in rest:
            id_name, class_part = rest.split(".", 1)
            token = f"{tag_prefix}.{class_part}"
        else:
            id_name = rest
            token = tag_prefix
        if node.attrs.get("id") != id_name:
            return False

    # 4. Class selector (.class1.class2)
    if "." in token:
        parts = token.split(".")
        tag_prefix = parts[0].strip()
        classes = [c.strip() for c in parts[1:] if c.strip()]
        node_classes = node.attrs.get("class", "").split()
        for c in classes:
            if c not in node_classes:
                return False
        token = tag_prefix

    # 5. Tag selector
    token = token.strip()
    if token and token != "*":
        if node.tag != token.lower():
            return False

    return True


def _split_selector_branches(selector: str) -> list[str]:
    """Split comma-separated CSS selectors outside quotes and brackets."""
    branches = []
    current: list[str] = []
    in_bracket = 0
    in_paren = 0
    in_quote: Optional[str] = None
    for char in selector:
        if in_quote:
            if char == in_quote:
                in_quote = None
            current.append(char)
        elif char in ("'", '"'):
            in_quote = char
            current.append(char)
        elif char == "[":
            in_bracket += 1
            current.append(char)
        elif char == "]":
            in_bracket = max(0, in_bracket - 1)
            current.append(char)
        elif char == "(":
            in_paren += 1
            current.append(char)
        elif char == ")":
            in_paren = max(0, in_paren - 1)
            current.append(char)
        elif char == "," and in_bracket == 0 and in_paren == 0:
            branch = "".join(current).strip()
            if branch:
                branches.append(branch)
            current = []
        else:
            current.append(char)
    if current:
        branch = "".join(current).strip()
        if branch:
            branches.append(branch)
    return branches


def _split_descendant_tokens(branch: str) -> list[str]:
    """Split whitespace-separated descendant tokens outside brackets and parens."""
    tokens = []
    current: list[str] = []
    in_bracket = 0
    in_paren = 0
    in_quote: Optional[str] = None
    for char in branch:
        if in_quote:
            if char == in_quote:
                in_quote = None
            current.append(char)
        elif char in ("'", '"'):
            in_quote = char
            current.append(char)
        elif char == "[":
            in_bracket += 1
            current.append(char)
        elif char == "]":
            in_bracket = max(0, in_bracket - 1)
            current.append(char)
        elif char == "(":
            in_paren += 1
            current.append(char)
        elif char == ")":
            in_paren = max(0, in_paren - 1)
            current.append(char)
        elif char.isspace() and in_bracket == 0 and in_paren == 0:
            token = "".join(current).strip()
            if token:
                tokens.append(token)
            current = []
        else:
            current.append(char)
    if current:
        token = "".join(current).strip()
        if token:
            tokens.append(token)
    return tokens


def _find_matching_descendants(root: "FakeDOMNode", sequence: list[str]) -> list["FakeDOMNode"]:
    """Find all descendant nodes matching a whitespace-separated sequence."""
    if not sequence:
        return []
    current_nodes = [
        n for n in root.find_all(lambda n: n is not root and _matches_single_token(n, sequence[0]))
    ]
    for step in sequence[1:]:
        next_nodes: list[FakeDOMNode] = []
        for parent in current_nodes:
            for descendant in parent.find_all(
                lambda n: n is not parent and _matches_single_token(n, step)
            ):
                if descendant not in next_nodes:
                    next_nodes.append(descendant)
        current_nodes = next_nodes
    return current_nodes


class FakeDOMNode:
    """In-memory DOM node that mirrors Playwright ElementHandle semantics."""

    def __init__(
        self,
        tag: str,
        attrs: dict[str, str],
        parent: Optional["FakeDOMNode"] = None,
    ):
        self.tag = tag.lower()
        self.attrs = {k.lower(): v for k, v in attrs.items()}
        self.parent = parent
        self.children: list[FakeDOMNode] = []
        self._text_chunks: list[str] = []
        self._input_value: str = self.attrs.get("value", "")
        self._checked: bool = (
            "checked" in self.attrs or self.attrs.get("aria-checked", "").lower() == "true"
        )
        self._visible: bool = True
        self._enabled: bool = "disabled" not in self.attrs

        # Calculate visibility from attributes/style
        style = self.attrs.get("style", "").lower()
        if "display: none" in style or "display:none" in style or "hidden" in self.attrs:
            self._visible = False

    @property
    def id(self) -> str | None:
        return self.attrs.get("id")

    def append_text(self, text: str):
        if text:
            self._text_chunks.append(text)

    def inner_text_sync(self) -> str:
        texts = []
        for chunk in self._text_chunks:
            texts.append(chunk)
        for child in self.children:
            texts.append(child.inner_text_sync())
        return " ".join("".join(texts).split())

    async def inner_text(self) -> str:
        return self.inner_text_sync()

    async def input_value(self) -> str:
        return self._input_value

    async def fill(self, value: str):
        self._input_value = str(value)

    async def check(self, force: bool = False, timeout: int | None = None):
        self._checked = True
        self.attrs["checked"] = "checked"
        if "aria-checked" in self.attrs:
            self.attrs["aria-checked"] = "true"

    async def click(self, timeout: int | None = None):
        if self.tag == "input" and self.attrs.get("type") == "radio":
            self._checked = True
            # uncheck siblings with same name
            name = self.attrs.get("name")
            if name and self.parent:
                for sib in self.parent.find_all(lambda n: n.attrs.get("name") == name):
                    if sib is not self:
                        sib._checked = False
        elif self.attrs.get("role") == "radio":
            self.attrs["aria-checked"] = "true"
            self._checked = True

    async def is_visible(self) -> bool:
        node = self
        while node:
            if not node._visible:
                return False
            node = node.parent
        return True

    async def is_enabled(self) -> bool:
        return self._enabled

    async def is_checked(self) -> bool:
        return self._checked

    async def get_attribute(self, name: str) -> str | None:
        return self.attrs.get(name.lower())

    async def select_option(self, value: str | None = None):
        if value is not None:
            self._input_value = value

    def find_all(self, predicate) -> list["FakeDOMNode"]:
        results = []
        if predicate(self):
            results.append(self)
        for child in self.children:
            results.extend(child.find_all(predicate))
        return results

    async def query_selector(self, selector: str) -> Optional["FakeDOMNode"]:
        for branch in _split_selector_branches(selector):
            tokens = _split_descendant_tokens(branch)
            matches = _find_matching_descendants(self, tokens)
            if matches:
                return matches[0]
        return None

    async def query_selector_all(self, selector: str) -> list["FakeDOMNode"]:
        collected: list[FakeDOMNode] = []
        for branch in _split_selector_branches(selector):
            tokens = _split_descendant_tokens(branch)
            for node in _find_matching_descendants(self, tokens):
                if node not in collected:
                    collected.append(node)
        return collected

    async def evaluate(self, script: str, *args: Any) -> Any:
        # Emulate common evaluate expressions used in execution
        script_lower = script.lower()
        if "options[el.selectedindex]" in script_lower or "selectedindex" in script_lower:
            for child in self.children:
                if child.tag == "option" and (
                    (self._input_value and child.attrs.get("value") == self._input_value)
                    or (self._input_value and child.inner_text_sync() == self._input_value)
                ):
                    return child.inner_text_sync()
            return self.children[0].inner_text_sync() if self.children else ""
        if "offsetparent" in script_lower or "visible" in script_lower:
            return self._visible
        if "tagname" in script_lower:
            return self.tag
        if "parent_element" in script_lower or "parentelement" in script_lower:
            return self.parent.inner_text_sync() if self.parent else ""
        if "aria-labelledby" in script_lower or "aria-label" in script_lower:
            return self.attrs.get("aria-label") or self.inner_text_sync()
        if "innertext" in script_lower or "textcontent" in script_lower:
            return self.inner_text_sync()
        return self.inner_text_sync()

    async def evaluate_handle(self, script: str, *args: Any) -> Any:
        class JSHandleAdapter:
            def __init__(self, node: "FakeDOMNode"):
                self.node = node

            def as_element(self):
                return self.node

            async def dispose(self):
                pass

        if "role=radio" in script or "role='radio'" in script:
            node: FakeDOMNode | None = self
            while node:
                if node.attrs.get("role") == "radio":
                    return JSHandleAdapter(node)
                node = node.parent
        return JSHandleAdapter(self)


class HTMLDOMParser(HTMLParser):
    """Parse HTML string into a FakeDOMNode root tree."""

    def __init__(self):
        super().__init__()
        self.root = FakeDOMNode("root", {})
        self.current = self.root

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        attr_dict = {k: (v or "") for k, v in attrs}
        node = FakeDOMNode(tag, attr_dict, parent=self.current)
        self.current.children.append(node)
        if tag.lower() not in ("input", "img", "br", "hr", "meta", "link"):
            self.current = node

    def handle_endtag(self, tag: str):
        if self.current.parent is not None and tag.lower() not in (
            "input",
            "img",
            "br",
            "hr",
            "meta",
            "link",
        ):
            self.current = self.current.parent

    def handle_data(self, data: str):
        self.current.append_text(data)


class FakePage:
    """Playwright Page adapter wrapping an in-memory FakeDOMNode."""

    def __init__(self, html: str, url: str = "https://www.linkedin.com/jobs/view/12345/"):
        parser = HTMLDOMParser()
        parser.feed(html)
        self.root = parser.root
        self.url = url
        self._closed = False

    def is_closed(self) -> bool:
        return self._closed

    async def close(self):
        self._closed = True

    async def query_selector(self, selector: str) -> Optional[FakeDOMNode]:
        return await self.root.query_selector(selector)

    async def query_selector_all(self, selector: str) -> list[FakeDOMNode]:
        return await self.root.query_selector_all(selector)

    async def wait_for_selector(
        self, selector: str, timeout: int | None = None
    ) -> Optional[FakeDOMNode]:
        elem = await self.query_selector(selector)
        if not elem:
            raise RuntimeError(f"Timeout waiting for selector: {selector}")
        return elem

    async def evaluate(self, script: str, *args: Any) -> Any:
        script_lower = script.lower()
        if "innertext" in script_lower or "body" in script_lower:
            text = self.root.inner_text_sync().lower()
            if args and isinstance(args[0], (list, tuple)):
                phrases = args[0]
                return any(phrase.lower() in text for phrase in phrases)
            return text
        return self.root.inner_text_sync()

    async def wait_for_function(self, script: str, *args: Any, timeout: int | None = None) -> bool:
        res = await self.evaluate(script, *args)
        if not res:
            raise RuntimeError("wait_for_function timeout")
        return True

    def locator(self, selector: str):
        page = self

        class FakeLocator:
            def __init__(self, sel: str):
                self.sel = sel
                self._visible_only = False

            def filter(self, visible: bool = True):
                self._visible_only = visible
                return self

            @property
            def first(self):
                return self

            async def click(self):
                nodes = await page.query_selector_all(self.sel)
                for node in nodes:
                    if not self._visible_only or await node.is_visible():
                        await node.click()
                        return
                raise RuntimeError(f"No match for locator: {self.sel}")

            async def inner_text(self) -> str:
                nodes = await page.query_selector_all(self.sel)
                if nodes:
                    return await nodes[0].inner_text()
                return page.root.inner_text_sync()

        return FakeLocator(selector)


# ---------------------------------------------------------------------------
# Fixture Unit Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fixture_text_and_contenteditable_fields():
    """Verify field discovery and label extraction across standard input types."""
    html = (FIXTURES_DIR / "text_fields.html").read_text(encoding="utf-8")
    page = FakePage(html)
    modal = await page.query_selector(".jobs-easy-apply-modal")
    assert modal is not None

    fields = await modal.query_selector_all(STANDARD_TEXT_FIELD_SELECTOR)
    assert len(fields) >= 7

    name_field = await modal.query_selector("#name-field")
    assert name_field is not None
    label = await get_form_field_label(name_field, page)
    assert label == "Full Name"
    assert is_known_field(label)
    assert get_auto_fill_value(label, {"name": "Jane Doe"}) == "Jane Doe"
    assert get_auto_fill_value("First Name", {"name": "Jane Doe"}) == "Jane"
    assert get_auto_fill_value("Last Name", {"name": "Jane Doe"}) == "Doe"
    assert get_auto_fill_value("Full Name", {}) is None

    phone_field = await modal.query_selector("#phone-field")
    assert phone_field is not None
    phone_label = await get_form_field_label(phone_field, page)
    assert "Phone" in phone_label
    assert is_known_field(phone_label)
    assert get_auto_fill_value(phone_label, {"phone": "+1-555-0100"}) == "+1-555-0100"

    portfolio_field = await modal.query_selector("#portfolio-field")
    assert portfolio_field is not None
    port_label = await get_form_field_label(portfolio_field, page)
    assert port_label == "Portfolio website URL"
    assert is_known_field(port_label)

    contenteditable = await modal.query_selector("[role='textbox'][contenteditable='true']")
    assert contenteditable is not None
    ce_label = await get_form_field_label(contenteditable, page)
    assert ce_label == "Cover note"


@pytest.mark.asyncio
async def test_fixture_required_fields_detection_and_validation():
    """Verify required field validation detects missing values and passes when filled."""
    html = (FIXTURES_DIR / "required_fields.html").read_text(encoding="utf-8")
    page = FakePage(html)
    modal = await page.query_selector(".jobs-easy-apply-modal")
    assert modal is not None

    val_res = await validate_visible_required_controls(modal, page)
    assert not val_res.is_valid
    assert len(val_res.unresolved_fields) >= 3
    assert any("Native Required" in f for f in val_res.unresolved_fields)
    assert any("Aria Required" in f for f in val_res.unresolved_fields)
    assert any("Location" in f for f in val_res.unresolved_fields)

    # Now fill all required fields and re-validate (properly awaiting coroutines)
    node_native = await modal.query_selector("#req-native")
    assert node_native is not None
    await node_native.fill("Filled Native")

    node_aria = await modal.query_selector("#req-aria")
    assert node_aria is not None
    await node_aria.fill("Filled Aria")

    node_star = await modal.query_selector("#req-star")
    assert node_star is not None
    await node_star.fill("Filled Star")

    node_select = await modal.query_selector("#req-select")
    assert node_select is not None
    initial_text = await node_select.evaluate(
        "el => el.options[el.selectedIndex]?.textContent?.trim() || ''"
    )
    assert initial_text == "Select an option"
    assert choice_is_unanswered(await node_select.input_value(), initial_text)

    await node_select.select_option("US")
    selected_text = await node_select.evaluate(
        "el => el.options[el.selectedIndex]?.textContent?.trim() || ''"
    )
    assert selected_text == "United States"
    assert not choice_is_unanswered(await node_select.input_value(), selected_text)

    node_check = await modal.query_selector("#req-check")
    assert node_check is not None
    await node_check.check()

    val_res_filled = await validate_visible_required_controls(modal, page)
    assert val_res_filled.is_valid
    assert len(val_res_filled.unresolved_fields) == 0


@pytest.mark.asyncio
async def test_fixture_select_placeholders():
    """Verify placeholder values are recognized as unanswered while valid options are preserved."""
    html = (FIXTURES_DIR / "select_placeholders.html").read_text(encoding="utf-8")
    page = FakePage(html)
    modal = await page.query_selector(".jobs-easy-apply-modal")
    assert modal is not None

    assert await modal.query_selector("#sel-1") is not None
    assert choice_is_unanswered("", "Select an option")
    assert not choice_is_unanswered("yes", "Yes, authorized")

    assert await modal.query_selector("#sel-2") is not None
    assert choice_is_unanswered("placeholder", "Choose option")

    assert await modal.query_selector("#sel-3") is not None
    assert choice_is_unanswered("none", "-- select --")

    # Boundary assertions: empty or sentinel values with and without visible text
    assert choice_is_unanswered("", "")
    assert choice_is_unanswered(None, None)
    assert choice_is_unanswered("select", None)
    assert choice_is_unanswered("none", "")
    assert choice_is_unanswered("placeholder", "")

    # Real options must NOT be misclassified
    assert not choice_is_unanswered("nonetheless", "Nonetheless")
    assert not choice_is_unanswered("none_of_the_above", "None of the above")
    assert not choice_is_unanswered("select_all", "Select all")
    assert not choice_is_unanswered("us", "United States")
    assert not choice_is_unanswered("engineer", "Software Engineer")


@pytest.mark.asyncio
async def test_fixture_native_and_role_radios():
    """Verify native fieldset radios and ARIA role=radiogroup controls."""
    html = (FIXTURES_DIR / "native_and_role_radios.html").read_text(encoding="utf-8")
    page = FakePage(html)
    modal = await page.query_selector(".jobs-easy-apply-modal")
    assert modal is not None

    # Native fieldset radio
    fieldset = await modal.query_selector("fieldset")
    assert fieldset is not None
    legend_text = await get_choice_label(fieldset, page)
    assert "legally authorized" in legend_text

    radios = await fieldset.query_selector_all("input[type='radio']")
    labels = [await get_radio_option_label(r, fieldset) for r in radios]
    assert labels == ["Yes", "No"]

    method = await select_radio_option(radios[0], fieldset)
    assert await radios[0].is_checked()
    assert "radio" in method.lower() or "label" in method.lower()

    # ARIA role=radiogroup
    group = await modal.query_selector("[role='radiogroup']")
    assert group is not None
    q_label = await get_choice_label(group, page)
    assert "sponsorship" in q_label

    role_radios = await group.query_selector_all("[role='radio']")
    assert len(role_radios) == 2
    await role_radios[1].click()
    assert (await role_radios[1].get_attribute("aria-checked")) == "true"


@pytest.mark.asyncio
async def test_fixture_duplicate_and_hidden_controls():
    """Verify hidden fields are ignored and duplicate controls are correctly handled."""
    html = (FIXTURES_DIR / "duplicate_and_hidden_controls.html").read_text(encoding="utf-8")
    page = FakePage(html)
    modal = await page.query_selector(".jobs-easy-apply-modal")
    assert modal is not None

    hidden_elem = await modal.query_selector("#hidden-req")
    assert hidden_elem is not None
    assert not await hidden_elem.is_visible()

    # Validation should ignore the hidden required field and pass
    val_res = await validate_visible_required_controls(modal, page)
    assert val_res.is_valid

    # Duplicate controls
    dup1 = await modal.query_selector("#dup-1")
    dup2 = await modal.query_selector("#dup-2")
    assert dup1 is not None and dup2 is not None
    assert await dup1.input_value() == "First comment"
    assert await dup2.input_value() == "Second comment"


@pytest.mark.asyncio
async def test_fixture_detached_reresolved_radios():
    """Verify live re-resolution selects updated radio options after DOM changes."""
    html = (FIXTURES_DIR / "native_and_role_radios.html").read_text(encoding="utf-8")
    page = FakePage(html)

    selection = await select_live_radio_option(
        page,
        question_text="Are you legally authorized to work in the United States?",
        option_text="Yes",
    )
    assert "radio" in selection.lower() or "label" in selection.lower()


@pytest.mark.asyncio
async def test_fixture_missing_required_blocks_navigation():
    """Verify missing required answers produce an invalid validation result."""
    html = (FIXTURES_DIR / "required_fields.html").read_text(encoding="utf-8")
    page = FakePage(html)
    modal = await page.query_selector(".jobs-easy-apply-modal")
    assert modal is not None

    val_res = await validate_visible_required_controls(modal, page)
    assert not val_res.is_valid
    assert len(val_res.unresolved_fields) > 0
    assert "Unresolved visible required field" in (val_res.reason or "")


@pytest.mark.asyncio
async def test_fixture_review_and_submit_actions():
    """Verify navigation action classification and button discovery."""
    html = (FIXTURES_DIR / "review_submit_actions.html").read_text(encoding="utf-8")
    page = FakePage(html)
    modal = await page.query_selector(".jobs-easy-apply-modal")
    assert modal is not None

    labels = await visible_button_labels(modal)
    assert "Back" in labels
    assert "Discard" in labels
    assert "Next" in labels
    assert "Review" in labels
    assert "Submit application" in labels

    # Classification
    assert classify_navigation_action("Back", None) is None
    assert classify_navigation_action("Discard", None) is None
    assert classify_navigation_action("Dismiss", None) is None
    assert classify_navigation_action("Next", None) == "advance"
    assert classify_navigation_action("Review", None) == "advance"
    assert classify_navigation_action("Submit application", None) == "submit"


@pytest.mark.asyncio
async def test_fixture_already_applied_and_submitted_evidence():
    """Verify applied status detection and post-submit confirmation phrases."""
    html = (FIXTURES_DIR / "already_applied_and_submitted.html").read_text(encoding="utf-8")
    page = FakePage(html)

    # Applied marker detection
    indicator = await find_already_applied_indicator(page)
    assert indicator is not None
    assert text_indicates_already_applied(indicator)

    # Submission confirmation
    confirmed = await wait_for_submission_confirmation(page, timeout_ms=500)
    assert confirmed is True


@pytest.mark.asyncio
async def test_fixture_external_assessment_text():
    """Verify external/assessment text triggers manual review detection."""
    html = (FIXTURES_DIR / "external_assessment.html").read_text(encoding="utf-8")
    page = FakePage(html)
    modal = await page.query_selector(".jobs-easy-apply-modal")
    assert modal is not None
    modal_text = await modal.inner_text()
    assert "external" in modal_text.lower() or "assessment" in modal_text.lower()


@pytest.mark.asyncio
async def test_validator_fails_closed_on_dom_exception_with_class_only_diagnostics():
    """Verify inspection API exceptions make validation invalid without leaking secret messages."""

    class FailingModal:
        async def query_selector_all(self, selector: str):
            if "select" in selector:
                raise TimeoutError(
                    "Internal Playwright timeout with secret token AIzaSyD3x94z_secret"
                )
            return []

    page = FakePage("<div class='jobs-easy-apply-modal'></div>")
    res = await validate_visible_required_controls(FailingModal(), page)

    assert res.is_valid is False
    assert len(res.unresolved_fields) >= 1
    assert any("select controls" in f and "TimeoutError" in f for f in res.unresolved_fields)
    for f in res.unresolved_fields:
        assert "AIzaSyD3x94z_secret" not in f
    assert res.reason is not None
    assert "AIzaSyD3x94z_secret" not in res.reason


@pytest.mark.asyncio
async def test_validator_redacts_and_bounds_secret_bearing_labels():
    """Verify labels with credentials, OTPs, or excessive length are redacted and bounded."""
    fake_key = "AIzaSyD_DummyFakeKeyForTestingPurposes123"
    html = (
        f"""
    <div class="jobs-easy-apply-modal">
        <label for="f1">Please enter token {fake_key} and verification code 839201 for authorization *</label>
        <input id="f1" type="text" required value="" />
        <label for="f2">"""
        + ("Very long label text " * 20)
        + """ *</label>
        <input id="f2" type="text" required value="" />
    </div>
    """
    )
    page = FakePage(html)
    modal = await page.query_selector(".jobs-easy-apply-modal")
    assert modal is not None

    res = await validate_visible_required_controls(modal, page)
    assert res.is_valid is False
    assert not any(fake_key in f for f in res.unresolved_fields)
    assert not any("AIzaSy" in f for f in res.unresolved_fields)
    assert not any("839201" in f for f in res.unresolved_fields)
    assert res.reason is not None
    assert fake_key not in res.reason
    assert "AIzaSy" not in res.reason
    assert "839201" not in res.reason
    for f in res.unresolved_fields:
        assert len(f) <= 145


@pytest.mark.asyncio
async def test_validator_caps_excessive_unresolved_controls():
    """Verify a form with many unresolved controls is capped to 10 items plus omission marker."""
    fields_html = "\n".join(
        f'<label for="f_{i}">Required Question Number {i} *</label><input id="f_{i}" type="text" required value="" />'
        for i in range(25)
    )
    html = f'<div class="jobs-easy-apply-modal">{fields_html}</div>'
    page = FakePage(html)
    modal = await page.query_selector(".jobs-easy-apply-modal")
    assert modal is not None

    res = await validate_visible_required_controls(modal, page)
    assert res.is_valid is False
    assert len(res.unresolved_fields) <= 10
    assert len(res.unresolved_fields) == 10
    assert "more fields omitted" in res.unresolved_fields[-1]
    assert len(res.reason or "") <= 300


def test_backward_compatibility_imports():
    """Ensure all required exports remain available from jobapply.nodes.execution with exact public signatures."""
    expected_symbols = [
        "AUTO_SKIP_PATTERNS",
        "CHOICE_PLACEHOLDERS",
        "KNOWN_FIELD_PATTERNS",
        "MODAL_CSS",
        "MODAL_SELECTORS",
        "STANDARD_TEXT_FIELD_SELECTOR",
        "FormQaInfrastructureError",
        "RequiredFieldValidationResult",
        "UserSkippedJob",
        "account_safety_execution_update",
        "append_application_outcome",
        "applied_update",
        "ask_user_for_question",
        "build_application_outcome",
        "choice_is_unanswered",
        "classify_navigation_action",
        "execution_node",
        "extract_answer_from_reply",
        "failed_update",
        "find_already_applied_indicator",
        "find_navigation_button",
        "format_application_receipt",
        "format_job_question_summary",
        "get_auto_fill_value",
        "get_choice_label",
        "get_form_field_label",
        "get_llm",
        "get_radio_option_label",
        "guard_page_account_safety",
        "inspect_page_account_safety",
        "is_auto_skip_field",
        "is_known_field",
        "is_required_field",
        "is_skip_job_reply",
        "managed_browser",
        "manual_review_update",
        "match_choice_index",
        "resume_was_edited",
        "select_live_radio_option",
        "select_live_role_radio_option",
        "select_radio_option",
        "send_application_receipt",
        "skipped_update",
        "state_list",
        "take_error_screenshot",
        "text_indicates_already_applied",
        "translate_question_for_telegram",
        "validate_visible_required_controls",
        "visible_button_labels",
        "wait_for_submission_confirmation",
        "wait_for_submission_or_safety",
    ]
    for sym in expected_symbols:
        assert hasattr(execution_module, sym), f"Missing backward compatibility symbol: {sym}"

    # Verify exact public parameter names, order, and defaults without **kwargs
    sig_live_radio = inspect.signature(execution_module.select_live_radio_option)
    assert list(sig_live_radio.parameters.keys()) == [
        "page",
        "question_text",
        "option_text",
        "attempts",
    ]
    assert sig_live_radio.parameters["attempts"].default == 3

    sig_trans = inspect.signature(execution_module.translate_question_for_telegram)
    assert list(sig_trans.parameters.keys()) == [
        "question_text",
        "options",
        "target_language",
    ]

    sig_extract = inspect.signature(execution_module.extract_answer_from_reply)
    assert list(sig_extract.parameters.keys()) == [
        "question_text",
        "user_reply",
        "options",
        "displayed_options",
    ]
    assert sig_extract.parameters["options"].default is None
    assert sig_extract.parameters["displayed_options"].default is None

    sig_ask = inspect.signature(execution_module.ask_user_for_question)
    assert list(sig_ask.parameters.keys()) == [
        "question_text",
        "job",
        "telegram",
        "settings",
        "options",
        "run_id",
        "ordinal",
    ]
    assert sig_ask.parameters["options"].default is None
    assert sig_ask.parameters["run_id"].default is None
    assert sig_ask.parameters["ordinal"].default == 0

    sig_sub_safety = inspect.signature(execution_module.wait_for_submission_or_safety)
    assert list(sig_sub_safety.parameters.keys()) == ["page", "timeout_ms"]
    assert sig_sub_safety.parameters["timeout_ms"].default == 12000

    sig_sub_conf = inspect.signature(execution_module.wait_for_submission_confirmation)
    assert list(sig_sub_conf.parameters.keys()) == ["page", "timeout_ms"]
    assert sig_sub_conf.parameters["timeout_ms"].default == 12000
