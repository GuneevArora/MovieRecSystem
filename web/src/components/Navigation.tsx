import { ArrowUpRight } from "lucide-react";
import { repoUrl } from "@/lib/project";

const links = [
  ["Demo", "#demo"],
  ["How it works", "#how"],
  ["Results", "#results"],
];

export function Navigation() {
  return (
    <nav className="liquid-glass fixed left-1/2 top-4 z-40 flex w-[calc(100%-32px)] max-w-6xl -translate-x-1/2 items-center justify-between rounded-full px-4 py-3 md:px-5">
      <a href="#home" className="text-sm font-semibold tracking-[-.03em]">
        GA<span className="text-[#8ea915]">.</span>
        <span className="ml-2 hidden text-black/40 sm:inline">/ Recommender</span>
      </a>
      <div className="hidden items-center gap-7 text-[11px] font-semibold uppercase tracking-[.16em] text-black/55 md:flex">
        {links.map(([label, href]) => (
          <a key={label} className="transition-colors hover:text-black" href={href}>
            {label}
          </a>
        ))}
      </div>
      <a
        href={repoUrl}
        target="_blank"
        rel="noreferrer"
        className="flex items-center gap-2 rounded-full bg-[#171816] px-4 py-2 text-xs font-semibold text-white transition-transform hover:scale-[1.03]"
      >
        Source <ArrowUpRight size={14} />
      </a>
    </nav>
  );
}
