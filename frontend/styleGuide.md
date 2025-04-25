# This is a NextJS Application.
# AgentSmyth Design System & Implementation Guidelines
## Color System Implementation
### Core Principles
1. Colors are defined in `globals.css` with base layer "@theme" directive defining default colors,
and [data-theme='dark'] defining dark mode colors.
2. By default, the application has data-theme set to dark.
3. All colors defined in tailwind variables should be derived from `globals.css`. If a color variables is not defined in this file, then it does not exist.

### Color Usage
Tailwind variables should be declared like this:
className="bg-(--color-agentBlue-500)"

Tailwind variables should NOT be declared like this:
className="bg-[--color-agentBlue-500]"

4. When creating new tailwind variables, do not use the "dark:" modifier.

### Example usage:
```
className="text-(--color-accent)"
className="bg-(--color-background) text-(--color-foreground)"
```

5. When choosing from colors, main colors should be either neutral, background, foreground, or agent colors.
Use accent colors for secondary text, buttons, highlighted moments, etc.
Use positive and negative colors for status indicators.

## Typography System
### Fonts
1. Headline (h1, h2, h3, h4, h5, h6) (Owners):
   ```tsx
   className="font-owners"
   ```
2. Secondary Font (h1, h2, h3, h4, h5, h6) (Citrine):
   ```tsx
   className="font-citrine"
   ```
3. Body (p) (sans-serif):
   Use default root font-family

### Weights
```tsx
font-{thin|extralight|light|normal|medium|semibold|bold|extrabold|black}
// Maps to: 100-900 weights
```
## Layout & Spacing
### Core Variables
1. Sidebar Width:
   ```css
   var(--spacing-left-sidebar-width): 260px
   ```
2. Transitions:
   ```css
   var(--transition-speed): 0.1s
   ```
### Animation Classes
1. Typing Animation:
   ```tsx
   className="animate-typing"
   ```
2. Thinking State:
   ```tsx
   className="animate-thinking-opacity"
   ```
3. Accordion:
   ```tsx
   className="animate-accordion-{up|down}"
   ```

## Best Practices
### Color Usage
1. Always use Tailwind classes over direct CSS
2. Use semantic color names (e.g., `bg-(--color-background)` not `bg-gray-100`)
3. Maintain consistent color usage across themes
4. Use CSS variables only when Tailwind classes aren't possible - for example, custom animations.

### Component Development
1. Follow existing patterns in the codebase
2. Use Tailwind's utility classes by default
3. Keep animations smooth with proper transitions

## File Structure
1. Global Styles: `src/globals.css`
2. Theme Config: `tailwind.config.ts`
3. Component Styles: Tailwind classes in component files

## Development Workflow
1. Start with Tailwind utilities
2. Use theme variables for dynamic values
3. Test in both light and dark modes
4. Verify color accessibility
5. Ensure smooth theme transitions

## Quality Checks
1. Verify correct Tailwind class usage
2. Check animation smoothness
3. Validate color contrast ratios
4. Ensure responsive design

---

## File structure

This is a NextJS Application which uses App Router directory structure.

Pages which are defined by a route should go in src/app directory. For example - dashboard, news-deck.

Base layer components (like shadcn components - for example, a button, an accordion, an input) should go in src/components/ui.

Composite or more complex components like a a combobox, or a complex card, can go in src/components directory.

Custom React hooks like useWindow or useViewportDimensions should go in src/hooks directory.