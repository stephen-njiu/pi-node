export default function Loading() {
  return (
    <div className="min-h-screen bg-linear-to-b from-[#0b1224] via-[#0c152c] to-[#0e1936] flex items-center justify-center">
      <div className="text-center">
        {/* Animated Logo/Spinner */}
        <div className="relative w-20 h-20 mx-auto mb-6">
          <div className="absolute inset-0 border-4 border-white/20 rounded-full"></div>
          <div className="absolute inset-0 border-4 border-transparent border-t-emerald-400 rounded-full animate-spin"></div>
          <div className="absolute inset-2 bg-[#14203f] rounded-full grid place-items-center">
            <span className="text-white font-bold text-sm">SG</span>
          </div>
        </div>

        {/* Loading Text */}
        <h2 className="text-xl font-semibold text-white mb-2">Savannah Gates</h2>
        <p className="text-white/60 text-sm">Please wait...</p>

        {/* Loading Dots Animation */}
        <div className="flex gap-1 justify-center mt-4">
          <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
          <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
          <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
        </div>
      </div>
    </div>
  );
}
