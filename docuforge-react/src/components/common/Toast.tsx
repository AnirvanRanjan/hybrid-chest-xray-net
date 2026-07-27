import type { Toast as ToastType } from '../../hooks/useToast';
import { COLORS } from '../../constants';

interface Props {
  toasts: ToastType[];
  onRemove: (id: number) => void;
}

export function ToastContainer({ toasts, onRemove }: Props) {
  if (toasts.length === 0) return null;

  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-2" style={{ maxWidth: 360 }}>
      {toasts.map(toast => (
        <div
          key={toast.id}
          className="flex items-center gap-3 px-4 py-3 rounded-xl animate-fade-in-up"
          style={{
            backgroundColor: COLORS.surfaceDark,
            border: `1px solid ${
              toast.type === 'success' ? COLORS.accentGreen :
              toast.type === 'error' ? COLORS.accentPink :
              COLORS.accentCyan
            }40`,
            boxShadow: '0 8px 32px rgba(0,0,0,0.3)',
          }}
        >
          <span style={{
            color: toast.type === 'success' ? COLORS.accentGreen :
                   toast.type === 'error' ? COLORS.accentPink :
                   COLORS.accentCyan,
            fontSize: '16px',
          }}>
            {toast.type === 'success' ? '✓' : toast.type === 'error' ? '✗' : 'ℹ'}
          </span>
          <p className="text-sm flex-1" style={{ color: COLORS.textPrimary }}>
            {toast.message}
          </p>
          <button
            onClick={() => onRemove(toast.id)}
            className="text-sm opacity-50 hover:opacity-100 transition-opacity"
            style={{ color: COLORS.textTertiary }}
          >
            ✕
          </button>
        </div>
      ))}
    </div>
  );
}
