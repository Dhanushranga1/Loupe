import { HashRouter, Routes, Route } from "react-router-dom";
import { Layout } from "./components/Layout";
import { Overview } from "./pages/Overview";
import { SymbolGraph } from "./pages/SymbolGraph";
import { Telemetry } from "./pages/Telemetry";
import { Conventions } from "./pages/Conventions";
import { Feedback } from "./pages/Feedback";

function App() {
  return (
    <HashRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Overview />} />
          <Route path="/graph" element={<SymbolGraph />} />
          <Route path="/telemetry" element={<Telemetry />} />
          <Route path="/conventions" element={<Conventions />} />
          <Route path="/feedback" element={<Feedback />} />
        </Route>
      </Routes>
    </HashRouter>
  );
}

export default App;
