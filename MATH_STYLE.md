# Math formatting for GitHub Markdown

Use GitHub-compatible math delimiters throughout this repository.

## Inline math

```markdown
$O(N^2)$
```

## Display math

```markdown
$$
\frac{\partial \mathcal L}{\partial h_l}
=
\frac{\partial \mathcal L}{\partial h_{l+1}}
\frac{\partial h_{l+1}}{\partial h_l}
$$
```

Avoid `\(...\)` and `\[...\]` in GitHub README files because GitHub Markdown may render them as escaped literal brackets rather than math.
