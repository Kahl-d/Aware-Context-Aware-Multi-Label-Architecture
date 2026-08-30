import { Outlet } from 'react-router';
import { NavBar } from './NavBar';

export function AppShell() {
  return (
    <div className="min-h-screen bg-white">
      <NavBar />
      <Outlet />
    </div>
  );
}
