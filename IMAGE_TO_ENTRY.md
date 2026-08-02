# Image-to-entry module plan

## Goal

`POST /image-to-entry` accepts one prescription image and returns a DentNode
`POST /entry/create` draft payload. It never creates or updates an entry.

## Processing pipeline

1. Authenticate the internal caller and validate the request shape.
2. Send the image bytes to the configured `LABY_VISION_MODEL` through OpenRouter.
3. Extract doctor, patient, case, work/product, FDI tooth numbers, shade, and all
   remaining readable instructions/notes.
4. Deterministically convert FDI numbers into DentNode's four-quadrant tooth
   chart and normalize DentNode enum values.
5. Return `entry_payload: {entry, patient, work}` with detected names and an
   explicit list of tenant-specific IDs that still need resolution.
6. Meter the model call as the `image_to_entry` feature.

## Contract and safety boundary

- Input images are base64 bytes; this service does not fetch user-provided URLs.
- The endpoint is protected by the existing `x-internal-key` / `x-internal-id`.
- Doctor and product IDs cannot be determined from pixels. The response keeps
  `productId` and `categoryId` null, reports detected names under `detected`, and
  sets `ready_to_create` to `false`. The DentNode backend must resolve those
  names within the authenticated lab before calling `/entry/create`.
- Extracted case IDs and dates are untrusted OCR output and must pass the normal
  DentNode create validation (including case-ID uniqueness).
- No persistence or entry-creation dependency exists in this module.

## Test plan

- Unit-test FDI-to-chart conversion and enum/default normalization.
- Endpoint-test authentication, request validation, success shape, metering,
  parse failures, and OpenRouter failures with the model call mocked.
- Before UI integration, run a labelled set of real lab forms and score doctor,
  patient, tooth, product, shade, and notes accuracy independently.
