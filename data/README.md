# Dataset

## Download and Decompress

The data zip may need to be downloaded manually by clicking on: [https://github.com/KEVINtoto/PSPGO/raw/main/data/data.tar.bz2](https://github.com/KEVINtoto/PSPGO/raw/main/data/data.tar.bz2).

Run the following command to decompress:
```
tar -jxv -f data.tar.bz2 -C data
```

## Description

The files in the `data` directory are:
 * `dgl_hetero_ppi_50_sim_50`: The `DGLHeteroGraph` contains two types of networks, a PPI network constructed from the raw network data obtained from STRING (sampling the 50 edges with the largest weights for each node), and a sequence similarity network constructed after sequence alignment by Diamond (also sampling the 50 edges with the largest weights for each node).
 * `network.fasta`: Sequence file of proteins on the network.
 * `interpro.npz`: The input features of the network nodes are obtained by [InterProScan](https://interproscan-docs.readthedocs.io/en/latest/index.html).
 * `pid2index.txt`: Convert the protein id to a network index.
 * `uniprot2string.txt`: Convert UniProt id to STRING id.

The files in the `bp/mf/cc` subdirectory are:
 * `*_go_ic.txt`: The information content of each GO term.
 * `*_*_go.txt`: The annotations of proteins.
 * `*_*.fasta`: The sequences of proteins.
