# Local host-embedded Tool extension (PLATFORM-PLUGIN-8)

This folder demonstrates **Mode B** from the Platform Plugin architecture: a
`ToolPlugin` implementation kept in the application source tree - no wheel, no
setuptools entry point, no `[tool.intergrax.plugin]` manifest.

## Workflow

1. Copy or adapt `local_prefix_echo_plugin.py` into your application package
   (scaffolded hosts use `<app_pkg>/extensions/`).
2. Implement the same public `ToolPlugin` contract as external packages.
3. Register explicitly from host composition, e.g. `register_tool_plugin(LocalPrefixEchoToolPlugin)`.
4. Pass host dependencies via `ToolWiringContext` (see `extras["echo_prefix"]`).
5. Run production qualification gates before enabling the bundle in production profiles.

See [`EXTENSION_AUTHOR_GUIDE.md`](../../../docs/project/technical/guides/EXTENSION_AUTHOR_GUIDE.md)
§16 (local quickstart) and the executable proof in
`tests/integration/platform_plugins/test_plugin8_dual_mode_tool_e2e.py`.
