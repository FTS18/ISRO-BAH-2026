# Behavioral Rules

- **Direct Analysis and Fixes**: Do not write temporary scratch scripts in the scratch folder or repeatedly run slow debug/profiling commands to test hypotheses. Analyze the codebase directly, understand the logic, and write clean, direct fixes where needed.
- **No Code Placeholders**: Never write placeholder code, partial implementations, or `# TODO` comments. Write complete, functional code.
- **Minimize Redundant File Reads**: Do not repeatedly read or grep the same files/symbols. Read the file once, extract the necessary context, and proceed.
- **Respect Existing Patterns & Dependencies**: Use the existing libraries, helper functions, and patterns in the repository. Do not introduce new library dependencies unless explicitly requested.
- **No Status Polling Loops**: Do not poll or query task/command statuses repeatedly. Run commands asynchronously and let the system trigger wakeups, or execute synchronously if fast.
- **Concise Communication**: Keep chat interactions brief, clear, and direct. Avoid repeating the user's instructions back to them at length or asking trivial questions.
