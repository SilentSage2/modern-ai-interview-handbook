# Math Style for GitHub Markdown

Use GitHub's native MathJax-compatible syntax.

## Display equations — required style

Use fenced `math` blocks:

````markdown
```math
q(x_t|x_{t-1})
=
\mathcal N(\sqrt{1-\beta_t}x_{t-1},\beta_t I).
```
````

This is preferred throughout this repository because it avoids Markdown parsing conflicts with characters such as `|`, `_`, `<`, and `>`.

## Inline equations

For simple inline math, `$...$` is acceptable:

```markdown
The attention complexity is $O(T^2)$.
```

If an inline expression contains Markdown-sensitive syntax, use GitHub's backtick math delimiter:

```markdown
$`p(y|x)`$
```

## Do not use

- `\[ ... \]`
- `\( ... \)`
- multiline `$$ ... $$`

The repository uses fenced `math` blocks for all display equations.


## Robust sequence-history notation

For both clarity and GitHub rendering robustness, prefer explicit index ranges:

```math
x_{1:t-1}
```

rather than compact shorthand such as `x_<t`, and:

```math
o_{0:t},
\qquad
a_{0:t-1}
```

rather than history shorthand based on less-than symbols.

The explicit range makes it immediately clear which observations/actions are included.
