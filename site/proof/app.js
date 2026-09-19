function dashboard() {
  return {
    tab: "hitters",
    search: "",
    minPA: 100,
    minBF: 150,

    hitterRows: [],
    pitcherRows: [],
    risersH: [],
    risersP: [],
    asOfRaw: "",

    hitterSort: { key: "pred_woba_proof", dir: "desc" },
    pitcherSort: { key: "pred_fip_proof", dir: "asc" },

    hitterCols: [
      { key: "Name", label: "Name" },
      { key: "Tm", label: "Tm" },
      { key: "Age", label: "Age" },
      { key: "PA_ytd", label: "PA" },
      { key: "pred_woba_prior", label: "Preseason" },
      { key: "pred_woba_proof", label: "ROS wOBA" },
      { key: "lo80", label: "lo 80%" },
      { key: "hi80", label: "hi 80%" },
      { key: "delta", label: "Δ vs preseason" },
      { key: "BB%", label: "BB%" },
      { key: "K%", label: "K%" },
      { key: "ISO", label: "ISO" },
      { key: "BABIP", label: "BABIP" },
    ],
    pitcherCols: [
      { key: "Name", label: "Name" },
      { key: "Tm", label: "Tm" },
      { key: "Age", label: "Age" },
      { key: "role", label: "Role" },
      { key: "IP", label: "IP" },
      { key: "FIP_ytd", label: "FIP ytd" },
      { key: "ERA_ytd", label: "ERA ytd" },
      { key: "pred_fip_prior", label: "Preseason FIP" },
      { key: "pred_fip_proof", label: "ROS FIP" },
      { key: "lo80", label: "lo 80%" },
      { key: "hi80", label: "hi 80%" },
      { key: "era_ros", label: "ROS ERA" },
      { key: "delta", label: "Δ vs preseason" },
    ],

    get asOfLabel() {
      if (!this.asOfRaw) return "…";
      const d = new Date(this.asOfRaw + "T00:00:00");
      return d.toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric" });
    },

    get filteredHitters() {
      let rows = this.hitterRows.filter(r => r.PA_ytd >= this.minPA);
      if (this.search) {
        const s = this.search.toLowerCase();
        rows = rows.filter(r => r.Name.toLowerCase().includes(s));
      }
      return this._sort(rows, this.hitterSort.key, this.hitterSort.dir);
    },

    get filteredPitchers() {
      let rows = this.pitcherRows.filter(r => r.BF_ytd >= this.minBF);
      if (this.search) {
        const s = this.search.toLowerCase();
        rows = rows.filter(r => r.Name.toLowerCase().includes(s));
      }
      return this._sort(rows, this.pitcherSort.key, this.pitcherSort.dir);
    },

    _sort(rows, key, dir) {
      const sorted = [...rows].sort((a, b) => {
        const va = a[key] ?? 0, vb = b[key] ?? 0;
        return va < vb ? -1 : va > vb ? 1 : 0;
      });
      return dir === "desc" ? sorted.reverse() : sorted;
    },

    sortHitters(key) {
      if (this.hitterSort.key === key) {
        this.hitterSort.dir = this.hitterSort.dir === "asc" ? "desc" : "asc";
      } else {
        this.hitterSort = { key, dir: key === "Name" || key === "Tm" ? "asc" : "desc" };
      }
    },
    sortPitchers(key) {
      if (this.pitcherSort.key === key) {
        this.pitcherSort.dir = this.pitcherSort.dir === "asc" ? "desc" : "asc";
      } else {
        this.pitcherSort = { key, dir: key === "Name" || key === "Tm" || key === "role" ? "asc" : "desc" };
      }
    },

    async init() {
      const [h, p, rh, rp] = await Promise.all([
        fetch("data/hitters.json").then(r => r.json()),
        fetch("data/pitchers.json").then(r => r.json()),
        fetch("data/risers_hitters.json").then(r => r.json()),
        fetch("data/risers_pitchers.json").then(r => r.json()),
      ]);
      this.hitterRows = h.rows;
      this.pitcherRows = p.rows;
      this.risersH = rh;
      this.risersP = rp;
      this.asOfRaw = h.meta.as_of;
      this._renderHittersChart();
      this._renderPitchersChart();
    },

    _renderHittersChart() {
      vegaEmbed("#hittersChart", {
        $schema: "https://vega.github.io/schema/vega-lite/v5.json",
        data: { values: this.risersH },
        width: "container",
        height: 380,
        mark: "bar",
        encoding: {
          x: { field: "delta", type: "quantitative", title: "ROS projection − preseason prior" },
          y: { field: "Name", type: "nominal", sort: "-x" },
          color: {
            field: "direction", type: "nominal", legend: null,
            scale: { domain: ["riser", "faller"], range: ["#c0392b", "#1e8449"] },
          },
          tooltip: [
            { field: "Name", type: "nominal" },
            { field: "delta", type: "quantitative", format: ".4f" },
          ],
        },
      }, { actions: false });
    },

    _renderPitchersChart() {
      vegaEmbed("#pitchersChart", {
        $schema: "https://vega.github.io/schema/vega-lite/v5.json",
        data: { values: this.risersP },
        width: "container",
        height: 380,
        mark: "bar",
        encoding: {
          x: { field: "delta", type: "quantitative", title: "ROS projection − preseason prior" },
          y: { field: "Name", type: "nominal", sort: "-x" },
          color: {
            field: "direction", type: "nominal", legend: null,
            scale: { domain: ["riser", "faller"], range: ["#c0392b", "#1e8449"] },
          },
          tooltip: [
            { field: "Name", type: "nominal" },
            { field: "delta", type: "quantitative", format: ".4f" },
          ],
        },
      }, { actions: false });
    },
  };
}

function fmt3(v) { return v != null ? v.toFixed(3) : "—"; }
function fmt2(v) { return v != null ? v.toFixed(2) : "—"; }
function fmt1(v) { return v != null ? v.toFixed(1) : "—"; }
function fmtPct(v) { return v != null ? (v * 100).toFixed(1) + "%" : "—"; }
function fmtDelta3(v) { return v != null ? (v >= 0 ? "+" : "") + v.toFixed(3) : "—"; }
function fmtDelta2(v) { return v != null ? (v >= 0 ? "+" : "") + v.toFixed(2) : "—"; }
