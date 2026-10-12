# Sanitization and local resources

Presidio loads the installed English spaCy model locally on CPU. It replaces detected
personal identifiers with stable placeholders across related fields. Detection is
best effort; the immutable `Sanitized[T]` wrapper attests that the policy ran.

Warm the configured resource with `POST /local-models/presidio/ensure` or
`POST /deployments/presidio/ensure`. Models load on demand. Worker-thread leases protect
active work, including after caller cancellation; idle cleanup clears owned references.

`assemble_application()` owns cold resource factories and controllers. Direct
sanitization calls use the process-local default assembly. Standalone controllers
can be given a registry explicitly. Configuration loading does not load weights.
