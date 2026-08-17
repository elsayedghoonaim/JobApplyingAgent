"""Execution decomposition package for Easy Apply automation."""

from jobapply.execution.controls import (
    get_choice_label,
    get_form_field_label,
    get_radio_option_label,
    select_live_radio_option,
    select_live_role_radio_option,
    select_radio_option,
    validate_visible_required_controls,
)
from jobapply.execution.navigation import (
    MODAL_CSS,
    MODAL_SELECTORS,
    classify_navigation_action,
    find_already_applied_indicator,
    find_navigation_button,
    text_indicates_already_applied,
    visible_button_labels,
    wait_for_submission_confirmation,
    wait_for_submission_or_safety,
)
from jobapply.execution.outcomes import (
    account_safety_execution_update,
    applied_update,
    failed_update,
    format_application_receipt,
    manual_review_update,
    send_application_receipt,
    skipped_update,
)
from jobapply.execution.planning import (
    AUTO_SKIP_PATTERNS,
    CHOICE_PLACEHOLDERS,
    KNOWN_FIELD_PATTERNS,
    STANDARD_TEXT_FIELD_SELECTOR,
    RequiredFieldValidationResult,
    choice_is_unanswered,
    get_auto_fill_value,
    is_auto_skip_field,
    is_known_field,
    is_required_field,
    is_skip_job_reply,
    match_choice_index,
)
from jobapply.execution.telegram_qa import (
    FormQaInfrastructureError,
    UserSkippedJob,
    ask_user_for_question,
    extract_answer_from_reply,
    format_job_question_summary,
    translate_question_for_telegram,
)

__all__ = [
    # planning
    "AUTO_SKIP_PATTERNS",
    "CHOICE_PLACEHOLDERS",
    "KNOWN_FIELD_PATTERNS",
    "STANDARD_TEXT_FIELD_SELECTOR",
    "RequiredFieldValidationResult",
    "choice_is_unanswered",
    "get_auto_fill_value",
    "is_auto_skip_field",
    "is_known_field",
    "is_required_field",
    "is_skip_job_reply",
    "match_choice_index",
    # controls
    "get_choice_label",
    "get_form_field_label",
    "get_radio_option_label",
    "select_live_radio_option",
    "select_live_role_radio_option",
    "select_radio_option",
    "validate_visible_required_controls",
    # navigation
    "MODAL_CSS",
    "MODAL_SELECTORS",
    "classify_navigation_action",
    "find_already_applied_indicator",
    "find_navigation_button",
    "text_indicates_already_applied",
    "visible_button_labels",
    "wait_for_submission_confirmation",
    "wait_for_submission_or_safety",
    # telegram_qa
    "FormQaInfrastructureError",
    "UserSkippedJob",
    "ask_user_for_question",
    "extract_answer_from_reply",
    "format_job_question_summary",
    "translate_question_for_telegram",
    # outcomes
    "account_safety_execution_update",
    "applied_update",
    "failed_update",
    "format_application_receipt",
    "manual_review_update",
    "send_application_receipt",
    "skipped_update",
]
