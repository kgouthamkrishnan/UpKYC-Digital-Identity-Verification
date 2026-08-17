
import os
import random
import time
from dotenv import load_dotenv

load_dotenv()

ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
FROM_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER")

TWILIO_CONFIGURED = bool(ACCOUNT_SID and AUTH_TOKEN and FROM_NUMBER)

OTP_LENGTH = 6
OTP_VALID_SECONDS = 300 


_otp_store = {}


def generate_otp():
    return str(random.randint(10**(OTP_LENGTH - 1), 10**OTP_LENGTH - 1))


def send_otp(phone_number):
    """
    Generate an OTP for the given phone number (E.164 format,
    e.g. "+919876543210"). Sends via real SMS if Twilio is configured,
    otherwise returns the OTP directly for simulated display.
    """
    otp = generate_otp()
    _otp_store[phone_number] = {
        "otp": otp,
        "expires_at": time.time() + OTP_VALID_SECONDS,
        "attempts": 0
    }

    if TWILIO_CONFIGURED:
        try:
            from twilio.rest import Client
            from twilio.base.exceptions import TwilioRestException
            client = Client(ACCOUNT_SID, AUTH_TOKEN)
            client.messages.create(
                body=f"Your KYC verification code is: {otp}. Valid for 5 minutes.",
                from_=FROM_NUMBER,
                to=phone_number
            )
            return {"success": True, "mode": "sms", "message": f"OTP sent to {phone_number}"}
        except Exception as e:
            pass

   
    return {
        "success": True,
        "mode": "simulated",
        "message": "SIMULATED: no real SMS was sent (no SMS gateway configured for this prototype).",
        "otp": otp
    }


def verify_otp(phone_number, entered_otp, max_attempts=3):
    """
    Check the entered OTP against the stored one for this number.
    Returns a dict with verified status and a reason.
    """
    record = _otp_store.get(phone_number)

    if record is None:
        return {"verified": False, "reason": "No OTP was sent to this number. Request a new one."}

    if time.time() > record["expires_at"]:
        del _otp_store[phone_number]
        return {"verified": False, "reason": "OTP expired. Request a new one."}

    record["attempts"] += 1
    if record["attempts"] > max_attempts:
        del _otp_store[phone_number]
        return {"verified": False, "reason": "Too many incorrect attempts. Request a new OTP."}

    if entered_otp.strip() == record["otp"]:
        del _otp_store[phone_number]  # one-time use - consume it on success
        return {"verified": True, "reason": "Mobile ownership verified via OTP"}
    else:
        remaining = max_attempts - record["attempts"]
        return {"verified": False, "reason": f"Incorrect OTP. {remaining} attempt(s) remaining."}


if __name__ == "__main__":
    number = input("Enter phone number (E.164 format, e.g. +919876543210): ").strip()
    result = send_otp(number)
    print(result)

    if result["mode"] == "simulated":
        print(f"\n[SIMULATED SMS] Your OTP is: {result['otp']}\n")

    if result["success"]:
        code = input("Enter the OTP: ").strip()
        verify_result = verify_otp(number, code)
        print(verify_result)