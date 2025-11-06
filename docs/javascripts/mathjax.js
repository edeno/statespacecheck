window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ['$', '$']],
    displayMath: [["\\[", "\\]"], ['$$', '$$']],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  },
  startup: {
    ready: () => {
      MathJax.startup.defaultReady();
      MathJax.startup.promise.then(() => {
        // Typeset any content that was already on the page
        MathJax.typesetPromise();
      });
    }
  }
};

document$.subscribe(() => {
  if (typeof MathJax !== 'undefined' && MathJax.typesetPromise) {
    MathJax.startup.output.clearCache?.();
    MathJax.typesetClear?.();
    MathJax.texReset?.();
    MathJax.typesetPromise();
  }
})
