/**
 * @file Animation loop helper with play/pause/step/reset, a steps-per-frame
 *       budget and automatic pausing while the widget is off-screen
 *       (many widgets per page must not all burn CPU at once).
 */

export class Animator {
  /**
   * @param {HTMLElement} el element whose visibility gates the animation
   * @param {{step: () => boolean|void, draw: () => void, stepsPerFrame?: number, onState?: (running: boolean) => void}} o
   *   step() advances the simulation by one step; return false to stop (e.g. final time reached).
   */
  constructor(el, o) {
    this.o = o;
    this.running = false;
    this.visible = true;
    this.stepsPerFrame = o.stepsPerFrame ?? 1;
    this._raf = 0;
    if (typeof IntersectionObserver !== 'undefined') {
      new IntersectionObserver((en) => {
        this.visible = en[0].isIntersecting;
        if (this.visible && this.running) this._loop();
      }).observe(el);
    }
  }
  _loop() {
    cancelAnimationFrame(this._raf);
    const frame = () => {
      if (!this.running || !this.visible) return;
      const t0 = performance.now();
      for (let k = 0; k < this.stepsPerFrame; k++) {
        if (this.o.step() === false) { this.pause(); break; }
        if (performance.now() - t0 > 30) break; // keep the page responsive
      }
      this.o.draw();
      if (this.running) this._raf = requestAnimationFrame(frame);
    };
    this._raf = requestAnimationFrame(frame);
  }
  play() { if (this.running) return; this.running = true; this.o.onState?.(true); this._loop(); }
  pause() { this.running = false; cancelAnimationFrame(this._raf); this.o.onState?.(false); }
  toggle() { if (this.running) this.pause(); else this.play(); }
  /** Single step + redraw. */
  stepOnce() { this.o.step(); this.o.draw(); }
}
