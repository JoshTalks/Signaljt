"""Exception hierarchy for the signaljt client."""


class SignaljtError(Exception):
    """Base class for all errors raised by this library."""


class NotLoggedInError(SignaljtError):
    """No API key could be resolved from arg, env, or the saved token."""


class AuthenticationError(SignaljtError):
    """The API rejected the provided key (401/403)."""


class ApiError(SignaljtError):
    """The API returned an unexpected response."""


class DownloadError(SignaljtError):
    """A transfer failed after retries."""


class ManifestError(SignaljtError):
    """The dataset manifest was missing or malformed."""
