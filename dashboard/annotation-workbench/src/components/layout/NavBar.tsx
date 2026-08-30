import { NavLink } from 'react-router';
import { ROUTES } from '../../constants/routes';

const navItems = [
  { path: ROUTES.EXPLORER, label: 'Data Explorer' },
  { path: ROUTES.EXPLORATION, label: 'Explore' },
  { path: ROUTES.INFERENCE, label: 'Inference' },
  { path: ROUTES.PAPER, label: 'AWARE Paper' },
];

export function NavBar() {
  return (
    <header className="border-b border-[var(--color-border)] bg-white sticky top-0 z-50">
      <div className="max-w-[1400px] mx-auto px-6 h-14 flex items-center gap-8">
        <NavLink to="/" className="flex items-center gap-2 no-underline">
          <span className="text-[15px] font-semibold tracking-tight text-gray-900">
            AWARE
          </span>
          <span className="text-[13px] text-gray-400 font-normal hidden sm:inline">
            Research Dashboard
          </span>
        </NavLink>

        <nav className="flex items-center gap-1 ml-auto">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `px-3 py-1.5 rounded-md text-[13px] font-medium no-underline transition-colors ${
                  isActive
                    ? 'bg-gray-100 text-gray-900'
                    : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  );
}
