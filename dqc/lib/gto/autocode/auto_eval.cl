#!/usr/bin/env clisp
;;;; Copyright (C) 2014-2018  Qiming Sun <osirpt.sun@gmail.com>

(load "gen-code.cl")

(gen-eval "auto_eval1.c"
  '("GTOval_rr"         (r dot r))
  '("GTOval_rrrr"       (r dot r r dot r))
  '("GTOval_iprr"       (nabla r dot r))
  '("GTOval_iprrrr"     (nabla r dot r r dot r))
  '("GTOval_ipip"       (nabla nabla))
  '("GTOval_ipipip"     (nabla nabla nabla))
  '("GTOval_laplrr"     (nabla dot nabla r dot r ))
  '("GTOval_laplrrrr"   (nabla dot nabla r dot r r dot r))
  '("GTOval_lapl"       (nabla dot nabla))
  '("GTOval_iplapl"     (nabla nabla dot nabla))
  '("GTOval_ipiplapl"   (nabla nabla nabla dot nabla))
  '("GTOval_ig"         (#C(0 1) g))
  '("GTOval_ipig"       (#C(0 1) nabla g))
  '("GTOval_sp"         (sigma dot p))
  '("GTOval_ipsp"       (nabla sigma dot p))
  '("GTOval_iprc"       (nabla rc))
  '("GTOval_ipr"        (nabla r))
)
