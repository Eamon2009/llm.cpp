import { useEffect, useRef } from "react";

export function useAutoScroll<T extends HTMLElement>(dependency: number) {
  const ref = useRef<T | null>(null);

  useEffect(() => {
    if (ref.current) {
      ref.current.scrollTo({ top: ref.current.scrollHeight, behavior: "smooth" });
    }
  }, [dependency]);

  return ref;
}
