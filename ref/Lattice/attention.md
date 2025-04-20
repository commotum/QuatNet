# Executive Summary: Quaternion Positional Encoding for Hierarchical Systems

The universe exhibits a profound hierarchical structure, where entities are simultaneously wholes and parts within larger systems—a concept encapsulated by Arthur Koestler's notion of the holon. From subatomic particles to biological organisms, social organizations, and even abstract constructs like books (book → chapters → paragraphs → sentences → words → characters), all things exist as self-contained units nested within greater wholes, forming what Koestler termed a holarchy. This dual nature—autonomous yet integrated—mirrors the parable of the two watchmakers, where stable intermediate forms (subassemblies) enable efficient assembly of complex systems, unlike fragile designs that collapse under interruption.

Our key finding is that quaternions, with their unique four-dimensional structure (w, x, y, z), offer an elegant and efficient way to encode this hierarchical essence in a transformer architecture.

## Natural Representation

Quaternions naturally capture position, rotation, and scale in a compact 4D vector, making them ideal for representing holons across diverse domains—be it 3D spatial data (e.g., scenes, cells), linguistic structures, or code hierarchies. Unlike traditional sequential positional encodings, which impose a 1D order, quaternions provide a multidimensional framework where:

- Scale (magnitude) reflects hierarchical depth (e.g., book = 1, chapter = 2)
- Rotation (orientation) encodes relative position within a level (e.g., chapter 1 of 12 as a 1/12 rotation)
- Position (vector components) ties it to a 4D context, free of linear bias

## Implementation

We propose integrating quaternion positional encoding into transformers by embedding each token's quaternion components into a $(n, d_{model})$ input tensor (e.g., $(3, 512)$ for "The cat sat"). Rather than expanding model dimensions or stacking components separately, we elegantly segment $d_{model} = 512$ into four equal 128D chunks - one for each quaternion component (w, x, y, z). This maintains the model's computational efficiency while preserving the quaternion's hierarchical encoding power.

Each chunk receives one quaternion component (w, x, y, z) summed with the corresponding portion of the word embedding. This preserves the quaternion's positional saliency, leverages the word embedding's inherent uniqueness for grouping, and fits seamlessly into a standard transformer without architectural changes.

The result is a model that respects the holarchic nature of data—treating tokens as autonomous holons while capturing their integrative relationships—unlocking new potential for processing hierarchical, multidimensional systems efficiently.


## Design Evolution

### Initial Concept
Inspired by the parable of the two watchmakers, we leveraged quaternions' 4D structure (w, x, y, z) to efficiently encode hierarchical information in transformers. Quaternions naturally represent position, rotation, and scale - making them ideal for modeling holons across domains like 3D scenes, linguistic structures, and code hierarchies.

### Approach 1: Component-wise Expansion
Our first attempt encoded each token's quaternion components (w, x, y, z) as a single vector in the transformer input. For a model with $d_{model}=512$, this expanded each token into a 2048D vector (4 × 512). While preserving quaternion detail, this approach had significant drawbacks:

- Quadrupled feature dimensions
- 16x increase in computational cost (parameters and attention scale with $d_{model}^2$)
- Excessive memory and runtime requirements

### Approach 2: Sequence Stacking 
We then tried stacking four 512D rows per token (e.g., $(12,512)$ for three tokens). While maintaining $d_{model}=512$, this raised new concerns:

- 4x longer sequence length
- 4 separate vectors per token versus a unified quaternion encoding
- Ambiguity surrounding effects of a disjoint representation

### Final Design: Segmented Embedding
Our final streamlined solution:

1. Maintain standard $(n,512)$ input shape (e.g., $(3,512)$ for three tokens)
2. Split $d_{model}=512$ into four 128D segments:
   - w: dimensions 0-127
   - x: dimensions 128-255  
   - y: dimensions 256-383
   - z: dimensions 384-511
3. Sum each quaternion component with its corresponding word embedding segment

Key benefits:
- Preserves computational efficiency (unchanged $d_{model}$)
- No sequence length inflation
- Maintains quaternion positional saliency in dedicated subspaces
- Natural component grouping via word embeddings
- Processes tokens as unified holons

This elegant solution enables efficient modeling of hierarchical systems while preserving quaternion integrity and transformer architecture compatibility.