import type { PageId } from '../state/navContext';
import type { ViewMode } from '../state/viewContext';

// Guided tour for the public snapshot (VITE_STATIC=1). Numbers are read off the
// recorded v2 snapshot; re-check them if the snapshot is re-recorded.
export type TourStep = {
  view: ViewMode;
  page: PageId;
  param?: string;
  title: string;
  body: string;
  // Element to scroll to and outline: first match of `selector`, optionally the one containing `text`.
  focus?: { selector: string; text?: string };
};

export const TOUR: TourStep[] = [
  {
    view: 'analyst',
    page: 'reference',
    title: 'What the model knows',
    body: 'The detector shipped with this labeled validation set: three civil aircraft classes, clear skies, U.S. airfields, no military aircraft. It is the only labeled data Vigil has.',
    focus: { selector: '[data-tour="known-classes"]' },
  },
  {
    view: 'analyst',
    page: 'collections',
    title: 'New imagery arrives',
    body: '192 unlabeled collections (one airfield on one date) came in after delivery. Vigil flags detections unlike anything in the validation set: normally 3.1%. Nine collections flag enough to need review.',
    focus: { selector: 'tr', text: 'Tirana' },
  },
  {
    view: 'analyst',
    page: 'collection',
    param: 'tirana-international-airport_2019-02-07',
    title: 'Tirana: something new',
    body: 'Clear skies, yet 18 of 66 detections were flagged (27%). Most are fighter jets, which v1 called small civil aircraft. An analyst labeled six of them as a new class.',
    focus: { selector: '.key-facts-line' },
  },
  {
    view: 'analyst',
    page: 'changes',
    title: 'What the feedback changed',
    body: 'From those six examples, Vigil re-sorted 17 detections: 8 at Tirana and 9 to inspect at six other airfields. The trained model is not modified; accuracy on the labeled control set is unchanged.',
    focus: { selector: '[data-tour="feedback-result"]' },
  },
  {
    view: 'analyst',
    page: 'health',
    title: 'Keep, patch, or retrain?',
    body: 'Four indicators the field can measure, each with its rule written out. Today it says patch: the new class covers these for now, and the rules state what would trigger a retrain.',
    focus: { selector: '.health-headline' },
  },
  {
    view: 'benchmark',
    page: 'overview',
    title: 'Independent check',
    body: "This view uses the dataset's ground truth. Detector recall (share of aircraft found) is 93% on the control set but 42% in snow, and 65% for military vs. 84% civil aircraft.",
    focus: { selector: '.batch-strip' },
  },
];
