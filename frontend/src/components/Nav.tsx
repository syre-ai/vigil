import { Activity, ClipboardList, FileText, FolderClock, GitCompareArrows, Layers, LayoutDashboard, ListChecks, Map as MapIcon, ScatterChart, ShieldCheck, Target } from 'lucide-react';
import type { ComponentType } from 'react';
import type { PageId } from '../state/navContext';
import { useChangesSummary, useSummary } from '../hooks/useApi';
import { useNav } from '../state/useNav';
import { useView } from '../state/useView';
import { ProtoBadge } from './ProtoBadge';

type NavItem = { id: PageId; label: string; icon: ComponentType<{ size?: number; strokeWidth?: number }> };

const BENCHMARK_ITEMS: NavItem[] = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard },
  { id: 'batch', label: 'Batch', icon: Layers },
  { id: 'review', label: 'Review queue', icon: ListChecks },
  { id: 'embedding', label: 'Embedding', icon: ScatterChart },
  { id: 'envelope', label: 'Tested envelope', icon: ShieldCheck },
  { id: 'eval', label: 'Evaluation', icon: Target },
  { id: 'map', label: 'Map', icon: MapIcon },
  { id: 'report', label: 'Report', icon: FileText },
];

const ANALYST_ITEMS: NavItem[] = [
  { id: 'reference', label: 'Validation set', icon: ShieldCheck },
  { id: 'collections', label: 'Collections', icon: FolderClock },
  { id: 'field-review', label: 'Review (field)', icon: ClipboardList },
  { id: 'changes', label: 'Changes', icon: GitCompareArrows },
  { id: 'health', label: 'Model health', icon: Activity },
  { id: 'report', label: 'Report', icon: FileText },
];

// "Changes" nav badge: how many detections a version /adapt built past v1 re-sorted into an
// analyst class (see pages/ChangesPage.tsx) — only shown once a version beyond v1 actually exists,
// and only while there's something to verify.
function ChangesNavBadge() {
  const summary = useSummary();
  const versions = summary.data?.versions ?? [];
  const hasAdaptedVersion = versions.length > 1;
  const a = hasAdaptedVersion ? versions[0]?.id : undefined;
  const b = hasAdaptedVersion ? summary.data?.current_version : undefined;
  const changes = useChangesSummary(a, b);
  if (!changes.data || !changes.data.total) return null;
  return (
    <span className="nav-badge mono" title={`${changes.data.total} detections re-sorted since ${a}. Verify them`}>
      {changes.data.total}
    </span>
  );
}

export function Nav() {
  const { route, goto } = useNav();
  const { view } = useView();
  const items = view === 'analyst' ? ANALYST_ITEMS : BENCHMARK_ITEMS;
  return (
    <nav className="sidebar">
      <ul className="nav-list">
        {items.map((item, i) => {
          const Icon = item.icon;
          // The Collections nav item stays highlighted while viewing one collection's detail page.
          const active = route.page === item.id || (item.id === 'collections' && route.page === 'collection');
          return (
            <li key={item.id}>
              <button type="button" className={`nav-item${active ? ' active' : ''}`} onClick={() => goto(item.id)}>
                <Icon size={16} strokeWidth={1.75} />
                <span className="nav-item-label">{item.label}</span>
                {item.id === 'changes' && <ChangesNavBadge />}
                <span className="nav-index">{i + 1}</span>
              </button>
            </li>
          );
        })}
      </ul>
      <div className="sidebar-footer">
        <ProtoBadge />
      </div>
    </nav>
  );
}
