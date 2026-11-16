# Shared app theme

Edit `static/css/app-theme.css` to change the shared interface palette, font
family, common corner radii, or focus colors. The defaults preserve the current
IVIVC input and results pages. The existing FDA logo assets remain in
`static/images/`; U1a does not replace or recolor them.

Load `app-theme.css` before an interface's own stylesheet. The IVIVC pages load
it before `styles.css`. The theme defines only `--app-*` custom properties on
`:root`, so it does not impose a layout or require study-design files. Both
stylesheets must be available when serving an IVIVC page.

The primary, hover, active, and foreground tokens control branded actions,
including header links, analysis/export buttons, help buttons, checkboxes, and
selected tabs. Neutral tokens cover common surfaces, headings, and borders.
Keep the focus and status tokens distinct from brand colors. Existing secondary
shades, specialized badges, and scientific plot colors remain local; this is a
shared foundation rather than a complete restyling of every component.

For example, change `--app-primary` in `app-theme.css` to update its consumers
together. Review its contrast with `--app-on-primary` and `--app-surface` when
choosing a new color. App-specific CSS may alias its component variables to the
shared tokens. Keep selector rules, responsive layouts, and interaction states
in the appropriate interface stylesheet.

## Branch rollout

1. Apply and commit U1a on `main`. It works without the study-design app.
2. Merge `main` into `design` to carry over the same theme foundation.
3. In U1b on `design`, load this theme in the study-design page and map its
   existing component variables to the shared tokens. Reuse the FDA logo asset
   while keeping each app's header layout appropriate to its navigation.

Do not apply U1a independently on both branches. The merge should carry its
history and shared files forward. The only U1a template changes are the theme
stylesheet links; design-specific inputs and result explanations stay intact.
