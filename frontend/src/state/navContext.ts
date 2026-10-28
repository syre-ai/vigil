import { createContext } from 'react';

export type PageId =
  | 'overview'
  | 'batch'
  | 'review'
  | 'embedding'
  | 'envelope'
  | 'eval'
  | 'map'
  | 'report'
  | 'collections'
  | 'collection'
  | 'field-review'
  | 'changes'
  | 'health'
  | 'reference';

export type Route = { page: PageId; batchId?: string; groupId?: string };

export type NavContextValue = {
  route: Route;
  goto: (page: PageId, param?: string) => void;
};

export const NavContext = createContext<NavContextValue | null>(null);
