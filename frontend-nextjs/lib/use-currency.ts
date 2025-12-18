'use client';

import { useEffect, useState } from 'react';

export type Currency = 'KES' | 'USD';

interface CurrencyInfo {
  currency: Currency;
  symbol: string;
  rate: number; // KES to USD rate (1 USD = X KES)
}

const KES_TO_USD_RATE = 145; // 1 USD = 145 KES (approximate)

export function useCurrency(): CurrencyInfo {
  const [currencyInfo, setCurrencyInfo] = useState<CurrencyInfo>({
    currency: 'USD',
    symbol: '$',
    rate: KES_TO_USD_RATE,
  });

  useEffect(() => {
    // Detect country via timezone or IP lookup
    async function detectCountry() {
      try {
        // Method 1: Check timezone
        const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
        if (timezone.includes('Africa/Nairobi')) {
          setCurrencyInfo({
            currency: 'KES',
            symbol: 'KES',
            rate: KES_TO_USD_RATE,
          });
          return;
        }

        // Method 2: IP-based geolocation (fallback)
        const response = await fetch('https://ipapi.co/json/', {
          signal: AbortSignal.timeout(3000), // 3 second timeout
        });
        
        if (response.ok) {
          const data = await response.json();
          if (data.country_code === 'KE') {
            setCurrencyInfo({
              currency: 'KES',
              symbol: 'KES',
              rate: KES_TO_USD_RATE,
            });
          } else {
            setCurrencyInfo({
              currency: 'USD',
              symbol: '$',
              rate: KES_TO_USD_RATE,
            });
          }
        }
      } catch (error) {
        // Default to USD on error
        setCurrencyInfo({
          currency: 'USD',
          symbol: '$',
          rate: KES_TO_USD_RATE,
        });
      }
    }

    detectCountry();
  }, []);

  return currencyInfo;
}

// Helper function to format price
export function formatPrice(kesPrice: number, currency: Currency, rate: number): string {
  if (currency === 'KES') {
    return `KES ${kesPrice.toLocaleString()}`;
  } else {
    const usdPrice = Math.round(kesPrice / rate);
    return `$${usdPrice.toLocaleString()}`;
  }
}
