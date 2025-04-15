MathJax = {
  section: 1,
  tex: {
    tagformat: {
      number: (n) => MathJax.config.section + '.' + n,
      id: (tag) => 'eqn-id:' + tag
    }
  },
  startup: {
    ready() {
      MathJax.startup.defaultReady();
      MathJax.startup.input[0].preFilters.add(({math}) => {
        if (math.inputData.recompile) {
          MathJax.config.section = math.inputData.recompile.section;
        }
      });
      MathJax.startup.input[0].postFilters.add(({math}) => {
        if (math.inputData.recompile) {
          math.inputData.recompile.section = MathJax.config.section;
        }
      });
    }
  }
}; 