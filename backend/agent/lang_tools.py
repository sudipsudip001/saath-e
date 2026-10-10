from langchain.tools import tool
import re, os
import asyncio
import smtplib
from loguru import logger
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from email.message import EmailMessage
search_tool = DuckDuckGoSearchRun()

load_dotenv()


@tool
def get_current_time() -> str:
    """Get current local time. Use when user asks time/date."""
    from datetime import datetime

    return datetime.now().isoformat()


_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")

# Ways people dictate punctuation out loud.
_SPOKEN_SYMBOLS = (
    (" at ", "@"),
    ("(at)", "@"),
    (" dot ", "."),
    ("(dot)", "."),
    (" underscore ", "_"),
    ("(underscore)", "_"),
    (" hyphen ", "-"),
    (" dash ", "-"),
    (" plus ", "+"),
)


def normalize_email(value: str) -> str:
    """Clean up a spoken/ASR-mangled email address.

    Handles dictated punctuation ("john at gmail dot com") and removes the stray
    spaces the ASR inserts between letters that were spelled out one by one.
    """
    if not value:
        return ""

    text = f" {value.strip().strip('.,;:!?')} ".lower()
    for spoken, symbol in _SPOKEN_SYMBOLS:
        text = text.replace(spoken, symbol)

    # People pause between spelled letters, so the ASR adds spaces.
    return re.sub(r"\s+", "", text)


def _send_email_sync(recipient: str, subject: str, body: str) -> str:
    """Send an email synchronously.

    Runs in a worker thread (see :func:`write_email`) so the blocking SMTP
    round-trip doesn't stall the real-time voice pipeline's event loop.
    """
    smtp_server = os.getenv("SMTP_SERVER", "")
    email_user = os.getenv("EMAIL_USER", "")
    email_password = os.getenv("EMAIL_PASSWORD", "")
    # int(os.getenv("SMTP_PORT", "")) used to raise ValueError when the port was
    # unset; fall back to the standard submission port instead.
    smtp_port = int(os.getenv("SMTP_PORT", "587") or 587)

    if not all([smtp_server, email_user, email_password]):
        return (
            f"MOCK EMAIL SENT TO {recipient} (No SMTP config found):\n"
            f"Subject: {subject}\nBody: {body}"
        )

    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = email_user
    msg["To"] = recipient

    try:
        with smtplib.SMTP(smtp_server, smtp_port, timeout=30) as server:
            server.starttls()
            server.login(email_user, email_password)
            server.send_message(msg)
        return f"Successfully sent email to {recipient}"
    except Exception as e:
        logger.exception("Failed to send email")
        return f"Failed to send email: {e}"


@tool
async def write_email(
    recipient: str, subject: str, body: str
):
    """Send an email to a recipient.

    Args:
        recipient: The recipient's email address.
        subject: The subject line of the email.
        body: The plain-text body of the email.
    """
    recipient = normalize_email(recipient)
    if not _EMAIL_RE.match(recipient):
        return f"INVALID_RECIPIENT '{recipient}'... ask user to repeat slowly..."
    return await asyncio.to_thread(_send_email_sync, recipient, subject, body)


@tool
def web_search(query: str) -> str:
    """Search the web to find accurate answers.

    Args:
        query: The question to search in the web to find answers.
    """
    try:
        return search_tool.invoke(query)
    except Exception as e:
        return f"An error occurred while searching the web: {str(e)}"


tools = [write_email, get_current_time, web_search]
