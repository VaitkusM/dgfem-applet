/**
 * @file Landing page: chapter cards, math and glossary pop-ups.
 */
import { CHAPTERS } from './chapters.js';
import { GLOSSARY } from './glossary.js';
import { renderMath } from './chapter.js';

const cards = document.getElementById('cards');
cards.innerHTML = CHAPTERS.map((c) => `
  <a class="card" href="${c.file}">
    <div class="num">${c.num === 'A' ? 'APPENDIX' : `CHAPTER ${c.num}`}</div>
    <h3>${c.title}</h3>
    <p>${c.blurb}</p>
  </a>`).join('');

const waitKatex = () => new Promise((res) => {
  let n = 0;
  const p = () => (window.renderMathInElement || n++ > 100 ? res() : setTimeout(p, 50));
  p();
});
waitKatex().then(() => renderMath(document.body));

let pop = null;
document.querySelectorAll('.term[data-g]').forEach((el) => {
  el.addEventListener('mouseenter', () => {
    const g = GLOSSARY[el.dataset.g];
    pop = document.createElement('div');
    pop.className = 'glossary-pop';
    pop.innerHTML = `<b>${g.term}</b>${g.def}`;
    document.body.appendChild(pop);
    renderMath(pop);
    const r = el.getBoundingClientRect();
    pop.style.left = `${window.scrollX + r.left}px`;
    pop.style.top = `${window.scrollY + r.bottom + 6}px`;
  });
  el.addEventListener('mouseleave', () => { pop?.remove(); pop = null; });
});
