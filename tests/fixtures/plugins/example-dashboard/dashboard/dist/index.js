(function () {
  const registry = window.__HERMES_PLUGINS__;
  const sdk = window.__HERMES_PLUGIN_SDK__;

  if (!registry || !sdk) {
    console.warn("[example-dashboard] Hermes plugin SDK is not available");
    return;
  }

  function ExampleDashboardPlugin() {
    return sdk.React.createElement(
      "div",
      { style: { padding: "1rem" } },
      "Example dashboard plugin loaded."
    );
  }

  registry.register("example", ExampleDashboardPlugin);
})();
