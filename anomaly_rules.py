def detect_rule_anomaly(
    person_count: int,
    motion_score: float,
    crowd_limit: int = 3,
    motion_threshold: float = 2.5,
) -> tuple[bool, list[str]]:
    reasons = []
    if person_count > crowd_limit:
        reasons.append("crowd")
    if motion_score > motion_threshold:
        reasons.append("high_motion")
    return bool(reasons), reasons
