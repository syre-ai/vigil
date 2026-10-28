import type { ReviewSource } from '../api/types';
import { ChipDrawerContent } from './ChipDrawerContent';
import { Modal } from './Modal';

export function ChipDrawerModal({ annId, source, onClose }: { annId: number; source?: ReviewSource; onClose: () => void }) {
  return (
    <Modal onClose={onClose} width={920}>
      <ChipDrawerContent annId={annId} source={source} />
    </Modal>
  );
}
