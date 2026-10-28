import { Map as MapLibreMap, Marker, NavigationControl, Popup } from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { useEffect, useRef, useState } from 'react';
import { InfoPopover } from '../components/InfoPopover';
import { QueryBoundary } from '../components/QueryState';
import { useMapPoints } from '../hooks/useApi';
import type { MapPoint } from '../api/types';
import { pct, score } from '../lib/format';

// Includes the reference-domain ids (ref_train/ref_calib) as well as the
// operational batch ids — /map plots every parent image, not just
// operational ones.
const DOMAIN_ORDER = ['ref_train', 'ref_calib', 'ref_holdout', 'off_nadir', 'foreign', 'haze', 'snow', 'military', 'synthetic'];

const DOMAIN_COLORS: Record<string, string> = {
  ref_train: '#94a3b8',
  ref_calib: '#64748b',
  ref_holdout: '#1f5aa6',
  haze: '#0891b2',
  snow: '#38bdf8',
  off_nadir: '#15803d',
  foreign: '#7c5cbf',
  military: '#dc2626',
  synthetic: '#d97706',
};

const DOMAIN_LABELS: Record<string, string> = {
  ref_train: 'Reference (train)',
  ref_calib: 'Reference (calibration)',
  ref_holdout: 'Control (same domain as reference)',
  haze: 'Cloud cover / haze',
  snow: 'Snow',
  off_nadir: 'High off-nadir',
  foreign: 'New geography (outside USA)',
  military: 'Rare targets present (military)',
  synthetic: 'External-domain stress test',
};

function colorFor(domain: string): string {
  return DOMAIN_COLORS[domain] ?? '#9aa3af';
}

function labelFor(domain: string): string {
  return DOMAIN_LABELS[domain] ?? domain;
}

function radiusFor(nChips: number): number {
  return Math.min(22, 4 + Math.sqrt(nChips) * 1.6);
}

function MapCanvas({ points }: { points: MapPoint[] }) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const [hover, setHover] = useState<{ x: number; y: number; point: MapPoint } | null>(null);

  const domains = Array.from(new Set(points.map((p) => p.domain))).sort((a, b) => {
    const ia = DOMAIN_ORDER.indexOf(a);
    const ib = DOMAIN_ORDER.indexOf(b);
    return (ia === -1 ? 999 : ia) - (ib === -1 ? 999 : ib);
  });

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = new MapLibreMap({
      container: containerRef.current,
      style: {
        version: 8,
        sources: {
          osm: {
            type: 'raster',
            tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
            tileSize: 256,
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
          },
        },
        layers: [{ id: 'osm', type: 'raster', source: 'osm' }],
      },
      center: [10, 30],
      zoom: 1.4,
      attributionControl: { compact: true },
    });
    map.addControl(new NavigationControl({ showCompass: false }), 'top-right');
    mapRef.current = map;
    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const markers: Marker[] = [];
    const attach = () => {
      for (const p of points) {
        const el = document.createElement('div');
        el.className = 'map-dot';
        const size = radiusFor(p.n_chips) * 2;
        el.style.width = `${size}px`;
        el.style.height = `${size}px`;
        el.style.background = colorFor(p.domain);
        const popup = new Popup({ offset: 8, closeButton: false }).setHTML(
          `<div class="map-popup"><strong>${p.airfield}</strong><br/>${p.country} · ${labelFor(p.domain)}<br/>n_chips ${p.n_chips}<br/>novelty ${pct(p.novelty_rate)}${
            p.drift_score != null ? `<br/>drift ${score(p.drift_score)}` : ''
          }</div>`,
        );
        el.addEventListener('mouseenter', () => setHover({ x: 0, y: 0, point: p }));
        el.addEventListener('mousemove', (ev) => {
          const rect = containerRef.current?.getBoundingClientRect();
          if (!rect) return;
          setHover({ x: (ev as MouseEvent).clientX - rect.left, y: (ev as MouseEvent).clientY - rect.top, point: p });
        });
        el.addEventListener('mouseleave', () => setHover(null));
        const marker = new Marker({ element: el }).setLngLat([p.lon, p.lat]).setPopup(popup).addTo(map);
        markers.push(marker);
      }
    };
    let loadHandler: (() => void) | undefined;
    if (map.isStyleLoaded()) {
      attach();
    } else {
      loadHandler = attach;
      map.once('load', loadHandler);
    }
    return () => {
      if (loadHandler) map.off('load', loadHandler);
      for (const m of markers) m.remove();
    };
  }, [points]);

  return (
    <div ref={containerRef} className="map-canvas" style={{ position: 'relative' }}>
      <div className="map-legend">
        <div className="map-legend-title">Domain</div>
        {domains.map((d) => (
          <div className="map-legend-row" key={d}>
            <span className="map-legend-swatch" style={{ background: colorFor(d) }} />
            {labelFor(d)}
          </div>
        ))}
        <div className="map-legend-size-row">
          <span className="map-legend-size-dot" style={{ width: 8, height: 8 }} />
          <span className="map-legend-size-dot" style={{ width: 16, height: 16 }} />
          <span>size = chip count</span>
        </div>
      </div>
      {hover && (
        <div className="map-hover-tooltip" style={{ left: hover.x + 14, top: hover.y + 8 }}>
          <strong>{hover.point.airfield}</strong>
          <br />
          {hover.point.country} · {labelFor(hover.point.domain)}
          <br />
          {hover.point.n_chips.toLocaleString()} chips · novelty {pct(hover.point.novelty_rate)}
          {hover.point.drift_score != null && (
            <>
              <br />
              drift {score(hover.point.drift_score)}
            </>
          )}
        </div>
      )}
    </div>
  );
}

export function MapPage() {
  const mapQuery = useMapPoints();
  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Map</h1>
          <div className="page-sub">
            Parent images by centroid of their aircraft annotations, colored by domain, sized by chip count.
            <InfoPopover label="About this map">
              lat/lon = centroid of the parent image&apos;s aircraft polygons from the public RarePlanes GeoJSON. Hover or click a point for novelty rate and drift score.
            </InfoPopover>
          </div>
        </div>
      </div>
      <div className="panel" style={{ padding: 0, overflow: 'hidden' }}>
        <QueryBoundary isLoading={mapQuery.isLoading} error={mapQuery.error} data={mapQuery.data} loadingLabel="Loading map…">
          {(data) => <MapCanvas points={data.points} />}
        </QueryBoundary>
      </div>
    </div>
  );
}
