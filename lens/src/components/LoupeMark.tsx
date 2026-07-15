/** The crosshair-loupe mark — reused as the sidebar wordmark glyph and as a small
 * accent on page headers. Matches public/favicon.svg's motif so the tab icon and
 * the in-app mark are the same shape, not two different logos. */
export function LoupeMark({ size = 18, className = "" }: { size?: number; className?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none" className={className}>
      <circle cx="14" cy="14" r="8.25" stroke="currentColor" strokeWidth="2.2" />
      <path d="M19.8 19.8L26 26" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" />
      <path d="M14 9.5V18.5" stroke="currentColor" strokeWidth="1.3" strokeOpacity="0.55" strokeLinecap="round" />
      <path d="M9.5 14H18.5" stroke="currentColor" strokeWidth="1.3" strokeOpacity="0.55" strokeLinecap="round" />
    </svg>
  );
}
