"""Retriever interfaces adapted from amazon-science/piperag@b6eb158.

Copyright Amazon.com, Inc. or its affiliates. Licensed under Apache-2.0.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Retriever(ABC):
    @property
    @abstractmethod
    def num_neighbours(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def neighbour_len(self):
        raise NotImplementedError

    @abstractmethod
    def retrieve_neighbours(self, chunks):
        raise NotImplementedError


class DummyRetriever(Retriever):
    def __init__(self, num_neighbours: int, neighbour_len: int):
        self._num_neighbours = num_neighbours
        self._neighbour_len = neighbour_len

    @property
    def num_neighbours(self):
        return self._num_neighbours

    @property
    def neighbour_len(self):
        return self._neighbour_len

    def retrieve_neighbours(self, chunks):
        import torch

        return torch.ones(
            (chunks.shape[0], self.num_neighbours, self.neighbour_len),
            dtype=torch.int64,
            device=chunks.device,
        )
