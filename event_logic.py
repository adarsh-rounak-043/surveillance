# event_logic.py

CRITICAL_EVENTS = {"Abuse","Arrest","Arson","Assault","Burglary","Explosion","Fighting","RoadAccidents","Robbery","Shooting","Shoplifting","Stealing","Vandalism"}

class Action:
    def __init__(self, raise_alert: bool = False):
        self.raise_alert = raise_alert

def decide_event_action(label: str,
                        confidence: float,
                        critical_thresh: float = 0.5) -> Action:
    """
    Decide if an alert should be raised for this label + confidence.
    """
    if label in CRITICAL_EVENTS and confidence >= critical_thresh:
        return Action(raise_alert=True)
    else:
        return Action(raise_alert=False)
