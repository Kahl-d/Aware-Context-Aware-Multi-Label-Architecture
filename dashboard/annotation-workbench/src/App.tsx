import { BrowserRouter, Routes, Route, Navigate } from 'react-router';
import { AppShell } from './components/layout/AppShell';
import { DataExplorerPage } from './pages/DataExplorer';
import { InferencePage } from './pages/Inference';
import { ExplorationPage } from './pages/Exploration';
import { PaperPage } from './pages/Paper';
import { ROUTES } from './constants/routes';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppShell />}>
          <Route path={ROUTES.EXPLORER} element={<DataExplorerPage />} />
          <Route path={ROUTES.INFERENCE} element={<InferencePage />} />
          <Route path={ROUTES.EXPLORATION} element={<ExplorationPage />} />
          <Route path={ROUTES.PAPER} element={<PaperPage />} />
          <Route path="/" element={<Navigate to={ROUTES.EXPLORER} replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
