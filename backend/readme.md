# RoofVision API — README

This document explains all available backend API endpoints for the RoofVision project and how to call them from a React frontend.

============================================================
BASE URL
============================================================
http://localhost:8000/

All routes below assume this base.

============================================================
AUTHENTICATION ENDPOINTS
============================================================

POST /login
Login with email + password (or your implementation).
Response: JWT token + user info.
React usage: Save token to localStorage.

GET /google/auth
Google OAuth callback endpoint.

POST /sign-out
Logs out the currently authenticated user.

GET /index
Test endpoint for development.

============================================================
CUSTOMER ENDPOINTS
============================================================

GET /customers/
Fetch all customers belonging to the logged-in agent.

POST /customers/
Create a new customer.
Example body:
{
"name": "John Doe",
"email": "john@example.com",
"phone": "123123123",
"address": "123 Main Street"
}

GET /customers/{id}/
Get a single customer.

PUT /customers/{id}/
PATCH /customers/{id}/
Update a customer.

DELETE /customers/{id}/
Remove a customer.

============================================================
HOUSE ENDPOINTS
============================================================

GET /houses/
Get houses belonging to customers owned by the agent.

POST /houses/
Create a house.
Body:
{
"customer": 1,
"address": "123 Roof St",
"description": "Needs repair"
}

GET /houses/{id}/
Get a single house.

PUT/PATCH /houses/{id}/
Update house.

DELETE /houses/{id}/
Delete house.

============================================================
HOUSE IMAGE ENDPOINTS
============================================================

POST /house-images/
Uploads a house image to Google Cloud Storage and stores URL.
Requires multipart/form-data with fields:

    file: (image file)
    house: (house ID)

    Example form data:
        file: someimage.jpg
        house: 3

GET /house-images/
Get all images the agent is allowed to view.

DELETE /house-images/{id}/
Delete a house image.

============================================================
AGENT LOGS ENDPOINTS
============================================================

GET /agent-logs/
Fetch activity logs such as: - viewed profile - uploaded an image - etc.

POST /agent-logs/
Create an activity log (mostly handled by backend).

============================================================
HOW TO CALL THESE ENDPOINTS IN REACT
============================================================

## Example: Login

fetch("http://localhost:8000/login", {
method: "POST",
headers: { "Content-Type": "application/json" },
body: JSON.stringify({ email, password })
})
.then(res => res.json())
.then(data => localStorage.setItem("token", data.token));

## Example: Authenticated GET

fetch("http://localhost:8000/customers/", {
headers: {
"Authorization": "Bearer " + localStorage.getItem("token")
}
})
.then(res => res.json())
.then(data => console.log(data));

## Example: Upload House Image

const form = new FormData();
form.append("file", fileInput.files[0]);
form.append("house", selectedHouseId);

fetch("http://localhost:8000/house-images/", {
method: "POST",
headers: {
"Authorization": "Bearer " + localStorage.getItem("token")
},
body: form
})
.then(res => res.json())
.then(data => console.log("Uploaded image:", data));

============================================================
NOTES
============================================================

- All endpoints require authentication except login + Google OAuth.
- Agent-only access: customers, houses, and images are filtered by current agent.
- House image upload automatically sends the file to Google Cloud Storage.
- The backend stores only the public image URL.
