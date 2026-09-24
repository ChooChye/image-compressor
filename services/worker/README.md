# Worker service

Queue consumer for CPU-heavy image optimization.

Flow:

1. Receive job ID.
2. Fetch source object.
3. Invoke `smartimg-core`.
4. Upload output and report.
5. Update job state.

Workers scale separately from the API.
