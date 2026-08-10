"""
Calibration: recovering what an image can't tell you on its own.

An image records pixels and relative colours. Turning those into physical
lengths, or into colours comparable across two cameras, needs something in the
frame whose true properties are known -- a target of measured size, a colour
checker with published patches -- or a rig whose behaviour was established
elsewhere. Recovering that relationship is what this package does.

One module per kind of calibration, because the kinds have almost nothing in
common except their bookkeeping:

  scale -- pixels per millimetre, from a target of known width.
  color -- not written yet. When it is, it lands beside scale.py rather than
           inside it: a colour correction is a method, a matrix, an offset and
           an illuminant, and pretending it's shaped like one number would
           distort both.

The bookkeeping they DO share -- which occurrences a calibration applies to,
where it came from, and how it's stored -- is `records.calibrations`, and it is
deliberately incurious about the payload. That split is the whole design: this
package knows what a calibration MEANS, that one knows who it applies TO.
"""
