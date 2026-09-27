"""MedLaya triage question presets for Laya's decision engine.

These presets are decision-support signals only -- not a diagnostic tool.
They classify and route patient-facing text (intake notes, portal messages,
symptom descriptions) into urgency levels, care pathways, and emergency
flags. Every output must sit in front of a human reviewer with clinical
judgment; validate accuracy and calibration on your own patient population
before relying on any threshold, and keep a human in the loop for anything
touching emergency / red-flag detection.
"""

from __future__ import annotations


def urgency_question() -> dict:
    """Return an ESI-inspired 5-level urgency question (``score`` type).

    The scale is adapted from the Emergency Severity Index for text-only
    triage (no vitals available), so each level carries a short example
    phrase to ground it from text alone.

    Criteria are ordered from least to most urgent, as Laya ``score``
    questions expect an ordinal scale ascending in severity.

    Returns:
        dict: A Laya-compatible typed question with ``type``, ``instructions``,
            and ``criteria`` keys.

    Example:
        >>> q = urgency_question()
        >>> q["type"]
        'score'
        >>> len(q["criteria"])
        5
    """
    return {
        "type": "score",
        "instructions": (
            "How urgent is this case? Rate from least to most urgent "
            "using the 5-level emergency severity scale. "
            "Base the rating only on what the text describes."
        ),
        "criteria": [
            (
                "non-urgent: routine issue needing no urgent resources, "
                "e.g. 'need a prescription refill next week, mild cold "
                "for 3 days'"
            ),
            (
                "less-urgent: minor issue needing one routine resource, "
                "e.g. 'sprained ankle yesterday, can walk, mild swelling'"
            ),
            (
                "urgent: needs several resources or same-day evaluation, "
                "e.g. 'fever 39C for 2 days with ear pain, not improving'"
            ),
            (
                "emergent: high-risk situation needing immediate workup, "
                "e.g. 'severe abdominal pain with vomiting blood, "
                "feeling faint'"
            ),
            (
                "resuscitation: needs immediate life-saving intervention, "
                "e.g. 'chest tightness with shortness of breath, "
                "getting worse since this morning'"
            ),
        ],
    }


def red_flag_question() -> dict:
    """Return an emergency red-flag question (``noul`` type).

    The question asks for a calibrated probability that the message
    describes a possible medical emergency (cardiac, respiratory,
    neurological, severe bleeding/trauma).

    When uncertain, err toward flagging. The output is a signal for
    clinician review, not an automatic action or a diagnosis.

    Returns:
        dict: A Laya-compatible typed question with ``type`` and
            ``instructions`` keys.

    Example:
        >>> q = red_flag_question()
        >>> q["type"]
        'noul'
    """
    return {
        "type": "noul",
        "instructions": (
            "Does this describe a possible medical emergency "
            "(cardiac, respiratory, neurological, severe "
            "bleeding/trauma)? If uncertain, err toward flagging. "
            "This is a signal for clinician review, not an automatic "
            "action or a diagnosis."
        ),
    }


def pathway_question(departments: dict | None = None) -> dict:
    """Return a care-pathway routing question (``choice`` type).

    Args:
        departments: Optional mapping of pathway name to a one-line
            description, so each deployment can supply its own routing
            targets. When ``None``, defaults to a generic four-way
            split: ``emergency`` / ``urgent_care`` / ``primary_care`` /
            ``self_care``.

    Returns:
        dict: A Laya-compatible typed question with ``type``,
            ``instructions``, and ``criteria`` keys.

    Example:
        >>> q = pathway_question()
        >>> sorted(q["criteria"].keys())
        ['emergency', 'primary_care', 'self_care', 'urgent_care']
        >>> custom = pathway_question(
        ...     {"cardiology": "heart-related symptoms",
        ...      "dermatology": "skin-related symptoms"}
        ... )
        >>> sorted(custom["criteria"].keys())
        ['cardiology', 'dermatology']
    """
    if departments is None:
        departments = {
            "emergency": (
                "symptoms suggesting a medical emergency needing "
                "immediate care"
            ),
            "urgent_care": (
                "needs same-day attention but not emergency care"
            ),
            "primary_care": (
                "routine issue that can be scheduled normally"
            ),
            "self_care": (
                "minor issue manageable with self-care and advice"
            ),
        }
    return {
        "type": "choice",
        "instructions": "Which care pathway should handle this case?",
        "criteria": dict(departments),
    }


def triage_questions(departments: dict | None = None) -> dict:
    """Return the full MedLaya triage schema for ``Router.predict()``.

    Combines :func:`urgency_question`, :func:`red_flag_question`, and
    :func:`pathway_question` into a single ``questions`` dict keyed by
    ``urgency``, ``red_flag``, and ``pathway``.

    Args:
        departments: Optional custom routing targets, passed through
            to :func:`pathway_question`.

    Returns:
        dict: Question schema ready to pass as the ``questions``
            argument of ``Router.predict()`` / ``Agent.predict()``.

    Example:
        >>> from medlaya.presets import triage_questions
        >>> from laya import Router  # doctest: +SKIP
        >>> router = Router(preload=True)  # doctest: +SKIP
        >>> state = (
        ...     "I've had chest tightness and shortness of breath "
        ...     "since this morning, it's getting worse."
        ... )
        >>> result = router.predict(state, triage_questions())  # doctest: +SKIP
        >>> result["answers"]["pathway"]["choice"]  # doctest: +SKIP
        'emergency'
        >>> result["answers"]["red_flag"]["noul"]  # doctest: +SKIP
        0.94
    """
    return {
        "urgency": urgency_question(),
        "red_flag": red_flag_question(),
        "pathway": pathway_question(departments),
    }
