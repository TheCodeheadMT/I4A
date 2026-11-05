"""Error envelope (spec 9.3): error.code, error.message, error.retryable; never stack traces or inputs."""
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, retryable: bool = False,
                 headers: dict | None = None, extra: dict | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.retryable = status, code, message, retryable
        self.headers, self.extra = headers, extra


def envelope(code: str, message: str, retryable: bool = False, request_id: str | None = None, **extra) -> dict:
    body = {"error": {"code": code, "message": message, "retryable": retryable, **extra}}
    if request_id:
        body["request_id"] = request_id
    return body


def not_found() -> ApiError:
    return ApiError(404, "NOT_FOUND", "Not found.")


def install(app) -> None:
    @app.exception_handler(ApiError)
    async def _api(_: Request, e: ApiError):
        return JSONResponse(envelope(e.code, e.message, e.retryable, **(e.extra or {})), e.status, headers=e.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, e: RequestValidationError):
        problems = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err.get("loc", ()) if x != "body")
            kind = "unknown field" if err.get("type") == "extra_forbidden" else err.get("msg", "invalid")
            problems.append(f"{loc}: {kind}")
        return JSONResponse(envelope("INVALID_FIELDS", "; ".join(problems) or "Invalid request."), 400)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, e: StarletteHTTPException):
        code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(e.status_code, "HTTP_ERROR")
        return JSONResponse(envelope(code, str(e.detail)), e.status_code)
