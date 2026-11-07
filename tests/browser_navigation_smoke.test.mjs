import assert from "node:assert/strict";
import test from "node:test";

let chromium;
try {
  ({ chromium } = await import("playwright"));
} catch {
  test("Studio browser navigation smoke", { skip: "Playwright is not installed in this runtime" }, () => {});
}

if (chromium) {
  test("Studio browser navigation smoke", async () => {
    const browser = await chromium.launch({ headless: true });
    const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
    const consoleErrors = [];
    page.on("console", (message) => { if (message.type() === "error") consoleErrors.push(message.text()); });
    try {
      await page.goto(process.env.STUDIO_URL ?? "http://127.0.0.1:8772", { waitUntil: "networkidle" });
      assert.equal(await page.locator("html").getAttribute("data-posenerf-bridge"), "ready");
      await page.getByLabel("Experiment").selectOption({ label: "stage_d_b1_coverage_192_a3_extended_12k · completed" });
      await page.waitForTimeout(500);
      assert.equal(await page.locator("#trainPreset option").count(), 192);
      assert.equal(await page.locator("#valPreset option").count(), 12);
      await page.getByLabel("Validation camera").selectOption({ label: "seq-02/frame-000082.color.png" });
      await page.getByRole("button", { name: "Jump to selected camera" }).click();

      const canvas = page.locator("#viewport canvas");
      const box = await canvas.boundingBox();
      assert.ok(box, "Three.js canvas is visible");
      const pose = () => page.locator("#poseMatrix").innerText();
      const target = () => page.locator("#orbitTarget").innerText();
      const beforeOrbit = await pose();
      await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
      await page.mouse.down(); await page.mouse.move(box.x + box.width / 2 + 100, box.y + box.height / 2 + 50); await page.mouse.up();
      assert.notEqual(await pose(), beforeOrbit, "left drag orbits the camera");
      assert.match(await target(), /distance: [\d.]+/, "OrbitControls target remains finite");

      const beforePan = await target();
      const beforePanPose = await pose();
      await page.getByLabel("Navigation").selectOption("pan");
      await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
      await page.mouse.down(); await page.mouse.move(box.x + box.width / 2 + 80, box.y + box.height / 2 + 30); await page.mouse.up();
      assert.notEqual(await pose(), beforePanPose, "configured left drag pans camera position");
      assert.notEqual(await target(), beforePan, "configured left drag pans the OrbitControls target");
      await page.getByLabel("Navigation").selectOption("orbit");

      const beforeZoom = await target();
      await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2); await page.mouse.wheel(0, -500);
      assert.notEqual(await target(), beforeZoom, "wheel changes camera distance");
      assert.ok(JSON.parse(await pose()).flat().every(Number.isFinite), "camera matrix remains finite");

      await page.getByRole("button", { name: "Frame scene" }).click();
      await page.getByLabel("Quality").selectOption("balanced");
      await page.getByRole("button", { name: "Render current camera" }).click();
      await page.waitForFunction(() => !document.querySelector("#render").disabled, null, { timeout: 120000 });
      assert.match(await page.locator("#renderMeta").innerText(), /"camera_pose"/);
      assert.deepEqual(consoleErrors, []);
    } finally {
      await browser.close();
    }
  });
}
