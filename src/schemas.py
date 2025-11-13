from pydantic import BaseModel, Field

class UserNameExtract(BaseModel):
    """This class is designed for user name extraction from the user input."""
    user_name: str = Field(
        description=(
            "Extract the user name from the user input. "
            "If the name is not found, simply return 'unknown'. "
            "Example: 'Tell me about Al Amin infos' -> user_name: 'Al Amin'"
        )
    )