"""
File: user_model.py
Description: User data model for handling user entities and authentication.
Dependencies: None (can be extended with SQLAlchemy, Pydantic, etc.)
Author: Sample Team

This is a sample file demonstrating proper code structure and documentation.
Students should replace this with their actual implementation.
"""


class User:
    """
    Represents a user entity in the system.
    
    Attributes:
        user_id (str): Unique identifier for the user.
        username (str): The user's username.
        email (str): The user's email address.
        role (str): The user's role (e.g., 'admin', 'merchant_user', 'consumer').
    
    Methods:
        validate_email(): Validates the email format.
        to_dict(): Converts the user object to a dictionary.
    """
    
