# Contributing

Thanks for helping improve Realtime ASD.

Useful contributions include:

- Reproducible bug reports with ROS topic names, message types, logs, and system
  details.
- Documentation fixes for installation, models, Docker, or ROS setup.
- Tests for pure Python utilities in `asd/` and `sdk/asd_sdk/`.
- Runtime improvements that reduce latency or make deployment less
  machine-specific.

Please keep issues and pull requests tied to real behavior. Synthetic activity
does not help the project; reproducible feedback does.

## Local Checks

```bash
python3 -m compileall asd config nodes sdk/asd_sdk tools visualization
```

More focused tests will be added as the project is split into smaller, easier to
test modules.
