# Study-design Flask frontend

The study-design interface has a separate Flask entry point and shares the
repository's Python environment with the IVIVC app. The IVIVC entry point,
templates, and numerical backend are unchanged.

## Run locally

From the repository root, with `requirements.txt` installed:

```bash
python design_app.py
```

Open `http://127.0.0.1:5001`. The separate port permits running the IVIVC app
on its usual port at the same time. Debug mode is off by default.

For a hosted installation, use the hosting service's normal WSGI configuration
with `design_app:app`, or import `app` from `design_app` as its application
object. The built-in development server is for local use. A future landing
page can link to the two applications without combining their entry points.

## F1: setup and request handling

- `GET /` serves the input and comparison workspace.
- `POST /review` accepts multipart form data and returns a JSON response.
  It validates the selected task, its active reference/environment choice,
  and the advanced numerical settings. Success includes server-rendered setup
  review HTML; errors include a message and an optional field identifier.
- **Review setup** checks these choices. File reading, previews, uncertainty
  interpretation, schedule generation, and numerical assessment are connected
  in later patches. A setup review is not an analysis result.
- **Calculate comparison** and report generation remain unavailable.

JavaScript is required for the workspace. It leaves the input form mounted,
including selected browser files, while a normal request runs to completion.
The displayed setup review changes only after a successful response. Validation,
network, and server errors preserve the earlier review. Edits during a request
are distinguished from the settings submitted in that request.

Only the current browser page holds the working form and displayed review.
Refreshing or closing it clears that state. There is no project/result store,
session storage, job queue, polling, progress reporting, or background worker.
The upload limit defaults to 25 MiB per request. Hosted request timeouts will
limit numerical calculations once those are connected.

The app factory `create_app(config=None)` supports ordinary Flask configuration
for deployment and testing. Server failures are logged; their details are not
inserted into browser error messages.

## Checks

```bash
python -m pytest tests/test_design_app.py
node tests/test_design_frontend.cjs
```

The frontend checks use Node's built-in test runner and a small DOM fixture;
they require no additional JavaScript packages. Existing numerical and IVIVC
tests should continue to run in the usual application environment.
