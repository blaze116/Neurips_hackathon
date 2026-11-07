"""The track rule.

This rule is normative. The organizer's intake applies the same one, and the track the checker
reports for a submission is the track it is scored in.

The competition has three nested tracks. Everything allowed in black-box is allowed in white-box, and
everything allowed in white-box is allowed in open.

- **black-box**: pipelines that neither require access to nor modify the model's internals
  (weights, activations, or logits).
- **white-box**: pipelines that may access and modify the model's states and logits.
- **open**: pipelines that may additionally modify the model's weights.

A submission is placed in the highest track any of its controls requires.

Detection reads the control's class, which requires no per-method table and no model. Three readings
come from the toolkit's declarations on that class: its base class, its `steer_access()`, and the
backend capabilities its `requirements()` asks for. A control defined outside the toolkit inherits
all three from the base class it subclasses.

Detection is a floor. The organizer may place a submission higher after reading the method card and
the code, and not lower. A control that declares less access than its steer step uses, e.g., an input
control that reads hidden states in `steer()` while declaring `FACTS`, breaks the toolkit's contract
and violates the rules.
"""
from steerability.algorithms.core.execution.contracts import Capability

from starterkit import TRACK_ORDER

#: The track each `Capability` atom requires, over every member of the toolkit's `Capability`.
ATOM_TRACK = {
    Capability.BEAM_PROPOSALS: "black-box",
    Capability.IN_PROCESS_TORCH: "white-box",
    Capability.INTERVENTION_SPECS: "white-box",
    Capability.PER_STEP_LOGIT_SPECS: "white-box",
    Capability.HIDDEN_CAPTURE: "white-box",
    Capability.GUIDED_DECODING: "white-box",
    Capability.SERVE_CHECKPOINT: "open",
    Capability.SERVE_LORA: "open",
}


def _alternative_track(alternative) -> tuple[str, Capability | None]:
    """The highest track over one alternative's atoms, and the atom that decided.

    An alternative with no atoms requires nothing beyond the session contract and is `black-box`,
    with no deciding atom.

    Args:
        alternative: One `Alternative` from a phase's requirements.

    Returns:
        `(track, atom)`; `atom` is None for an alternative with no atoms and for one whose atoms are
        all black-box.
    """
    track, deciding = "black-box", None
    for atom in sorted(alternative.atoms, key=lambda atom: atom.value):
        candidate = ATOM_TRACK[atom]
        if TRACK_ORDER[candidate] > TRACK_ORDER[track]:
            track, deciding = candidate, atom
    return track, deciding


def _phase_track(alternatives) -> tuple[str, Capability | None]:
    """The lowest track over a phase's alternatives, and the atom that decided.

    A backend satisfies a phase by satisfying any one of its alternatives. The phase requires only
    the lowest track among them. A phase with no alternatives is `black-box`.

    Args:
        alternatives: The `Alternative` tuple of one phase.

    Returns:
        `(track, atom)`; `atom` is None when the deciding alternative needs no atom above black-box.
    """
    lowest: tuple[str, Capability | None] | None = None
    for alternative in alternatives:
        reading = _alternative_track(alternative)
        if lowest is None or TRACK_ORDER[reading[0]] < TRACK_ORDER[lowest[0]]:
            lowest = reading
    return lowest if lowest is not None else ("black-box", None)


def control_track(control) -> tuple[str, str]:
    """The track one control requires, and a one-phrase reason.

    The highest of three readings of the control's class decides:

    1. its category, from the toolkit base class it subclasses: a `StructuralControl` is `open`
       (it changes weights) and a `StateControl` is `white-box`
    2. its steer-time access: `steer_access()` of `CAPTURE` or above is `white-box`
    3. its generate-time requirements: for each of the `generate` and `score` phases, the phase's
       track is the lowest track over its alternatives, and an alternative's track is the highest
       track over its atoms

    Args:
        control: An instantiated control.

    Returns:
        `(track, reason)`, where `reason` is `"structural control"`, `"state control"`,
        `"steer access CAPTURE"`, `"requires per_step_logit_specs"`, or
        `"prompt or generation-level only"`.
    """
    from steerability.algorithms.core.execution.access import ModelAccess
    from steerability.algorithms.state_control.base import StateControl
    from steerability.algorithms.structural_control.base import StructuralControl

    if isinstance(control, StructuralControl):
        readings = [("open", "structural control")]
    elif isinstance(control, StateControl):
        readings = [("white-box", "state control")]
    else:
        readings = [("black-box", "prompt or generation-level only")]

    access = control.steer_access()
    if access >= ModelAccess.CAPTURE:
        readings.append(("white-box", f"steer access {access.name}"))

    requirements = control.requirements()
    for alternatives in (requirements.generate, requirements.score):
        track, atom = _phase_track(alternatives)
        reason = f"requires {atom.value}" if atom is not None else "prompt or generation-level only"
        readings.append((track, reason))

    return max(readings, key=lambda reading: TRACK_ORDER[reading[0]])


def control_tracks(controls) -> list[dict]:
    """One entry per control, enabled or not, in pipeline order.

    Disabled controls are listed and do not count. Only the enabled controls decide the bundle's
    track.

    Args:
        controls: Instantiated controls in pipeline order.

    Returns:
        `[{"method", "enabled", "track", "reason"}, ...]`.

    Raises:
        RegistryError: If a control's class is not registered. A control defined outside the toolkit
            registers itself with `register_method` at import.
    """
    from steerability.algorithms.core.registry import method_key_for

    entries = []
    for control in controls:
        track, reason = control_track(control)
        entries.append({
            "method": method_key_for(type(control)),
            "enabled": bool(getattr(control, "enabled", True)),
            "track": track,
            "reason": reason,
        })
    return entries


def track_of(controls) -> str:
    """The highest track over the enabled controls; `black-box` when there are none.

    Args:
        controls: Instantiated controls.

    Returns:
        One of `TRACKS`.
    """
    tracks = [
        control_track(control)[0]
        for control in controls
        if getattr(control, "enabled", True)
    ]
    if not tracks:
        return "black-box"
    return max(tracks, key=lambda track: TRACK_ORDER[track])
