import { useState } from 'react';
import { GuideDrawer } from './components/GuideDrawer';
import { Nav } from './components/Nav';
import { TopBar } from './components/TopBar';
import { BatchPage } from './pages/BatchPage';
import { ChangesPage } from './pages/ChangesPage';
import { ModelHealthPage } from './pages/ModelHealthPage';
import { CollectionPage } from './pages/CollectionPage';
import { Collections } from './pages/Collections';
import { Embedding } from './pages/Embedding';
import { EnvelopePage } from './pages/EnvelopePage';
import { Evaluation } from './pages/Evaluation';
import { FieldReview } from './pages/FieldReview';
import { MapPage } from './pages/MapPage';
import { Overview } from './pages/Overview';
import { ReferencePage } from './pages/ReferencePage';
import { Report } from './pages/Report';
import { ReviewQueue } from './pages/ReviewQueue';
import { NavProvider } from './state/nav';
import { useNav } from './state/useNav';
import { ToastProvider } from './state/toast';
import { ViewProvider } from './state/view';
import { ErrorBoundary } from './ErrorBoundary';

function Routed() {
  const { route } = useNav();
  switch (route.page) {
    case 'overview':
      return <Overview />;
    case 'batch':
      return <BatchPage />;
    case 'review':
      return <ReviewQueue />;
    case 'embedding':
      return <Embedding />;
    case 'envelope':
      return <EnvelopePage />;
    case 'eval':
      return <Evaluation />;
    case 'map':
      return <MapPage />;
    case 'report':
      return <Report />;
    case 'collections':
      return <Collections />;
    case 'collection':
      return <CollectionPage />;
    case 'field-review':
      return <FieldReview />;
    case 'changes':
      return <ChangesPage />;
    case 'health':
      return <ModelHealthPage />;
    case 'reference':
      return <ReferencePage />;
    default:
      return <Overview />;
  }
}

function RoutedWithBoundary() {
  const { route } = useNav();
  // Keyed by page so a crash on one page doesn't stick around after
  // navigating away from it.
  return (
    <ErrorBoundary key={route.page}>
      <Routed />
    </ErrorBoundary>
  );
}

function App() {
  const [guideOpen, setGuideOpen] = useState(false);
  return (
    <ViewProvider>
      <ToastProvider>
        <NavProvider>
          <div className="app-shell">
            <TopBar onGuideOpen={() => setGuideOpen(true)} />
            <div className="shell">
              <Nav />
              <main className="main">
                <RoutedWithBoundary />
              </main>
            </div>
          </div>
          <GuideDrawer open={guideOpen} onClose={() => setGuideOpen(false)} />
        </NavProvider>
      </ToastProvider>
    </ViewProvider>
  );
}

export default App;
