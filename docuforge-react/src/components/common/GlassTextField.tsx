import React, { useState, useRef, forwardRef } from 'react';

interface Props {
  label?: string;
  hint?: string;
  prefixIcon?: React.ReactNode;
  suffixIcon?: React.ReactNode;
  value?: string;
  onChange?: (e: React.ChangeEvent<HTMLTextAreaElement | HTMLInputElement>) => void;
  onFocus?: () => void;
  onBlur?: () => void;
  type?: string;
  integerOnly?: boolean;
  maxLines?: number;
  minLines?: number;
  className?: string;
  name?: string;
  id?: string;
}

export const GlassTextField = forwardRef<HTMLTextAreaElement | HTMLInputElement, Props>(
  (
    {
      label,
      hint,
      prefixIcon,
      suffixIcon,
      value,
      onChange,
      onFocus,
      onBlur,
      type = 'text',
      integerOnly = false,
      maxLines = 1,
      minLines,
      className = '',
      name,
      id,
    },
    ref
  ) => {
    const [isFocused, setIsFocused] = useState(false);
    const internalRef = useRef<HTMLTextAreaElement | HTMLInputElement>(null);
    const inputRef = (ref as React.RefObject<HTMLTextAreaElement | HTMLInputElement>) || internalRef;

    const handleFocus = () => {
      setIsFocused(true);
      onFocus?.();
    };

    const handleBlur = () => {
      setIsFocused(false);
      onBlur?.();
    };

    const inputStyle: React.CSSProperties = {
      width: '100%',
      padding: '16px 20px',
      fontSize: '15px',
      fontWeight: 400,
      color: '#F0F0F5',
      backgroundColor: 'rgba(255,255,255,0.08)',
      border: `1px solid ${isFocused ? 'rgba(0,212,255,0.5)' : 'rgba(255,255,255,0.15)'}`,
      borderRadius: '16px',
      outline: 'none',
      transition: 'all 0.3s ease',
      boxShadow: isFocused ? '0 0 16px 2px rgba(0,212,255,0.15)' : 'none',
      fontFamily: "'Inter', sans-serif",
    };

    const InputComponent = maxLines > 1 ? 'textarea' : 'input';

    return (
      <div className={`flex flex-col gap-2 ${className}`}>
        {label && (
          <label className="text-sm font-medium" style={{ color: 'rgba(240,240,245,0.7)', paddingLeft: '4px' }}>
            {label}
          </label>
        )}
        <div className="relative">
          {prefixIcon && (
            <div className="absolute left-4 top-1/2 -translate-y-1/2 pointer-events-none" style={{ color: 'rgba(240,240,245,0.5)' }}>
              {prefixIcon}
            </div>
          )}
          <InputComponent
            ref={inputRef as any}
            className={prefixIcon ? 'pl-12' : ''}
            style={inputStyle}
            placeholder={hint}
            value={value}
            onChange={onChange}
            onFocus={handleFocus}
            onBlur={handleBlur}
            type={integerOnly ? 'number' : type}
            {...(maxLines > 1 ? { rows: minLines || maxLines } : {})}
            name={name}
            id={id}
          />
          {suffixIcon && (
            <div className="absolute right-3 top-1/2 -translate-y-1/2">
              {suffixIcon}
            </div>
          )}
        </div>
      </div>
    );
  }
);

GlassTextField.displayName = 'GlassTextField';
