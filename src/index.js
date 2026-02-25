/**
 * meljs — Pure JavaScript mel spectrogram and audio feature extraction.
 *
 * @module meljs
 */

// Core classes
export {
  MelSpectrogram,
  IncrementalMelSpectrogram,
} from './mel.js';

// Low-level building blocks
export {
  hzToMel,
  melToHz,
  createMelFilterbank,
  createPaddedHannWindow,
  precomputeTwiddles,
  fft,
} from './mel.js';

// Constants
export { MEL_CONSTANTS } from './mel.js';

// Legacy aliases (parakeet.js compat)
export { JsPreprocessor, IncrementalMelProcessor } from './mel.js';

// Parakeet-specific variant processors (current + optimization candidates)
export {
  PARAKEET_MEL_VARIANTS,
  ParakeetCurrentMelProcessor,
  ParakeetMelProcessorPr74,
  ParakeetMelProcessorPr75,
  ParakeetMelProcessorPr84,
  ParakeetIncrementalMelProcessor,
  createParakeetMelProcessor,
} from './parakeet-variants.js';
